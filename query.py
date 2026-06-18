import argparse
import json
import textwrap
from typing import Any, Dict, List

from utils import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    logger,
    retry,
    get_gemini_client,
    get_pinecone_index,
)


# ── Query Rephraser ───────────────────────────────────────────────────────────

_REPHRASE_PROMPT = textwrap.dedent(
    """\
    You are a defect search specialist at a SaaS CRM company.
    Rephrase the user's input into an optimised query for vector-similarity search over a defect knowledge base.

    STRICT RULES:
    1. PRESERVE every specific feature name, module name, product name, technical keyword, and error
       description the user mentioned — do NOT substitute, paraphrase, or drop them.
    2. PRESERVE the user's core intent — failures, crashes, wrong behaviour, missing data, slowness, etc.
    3. CONVERT questions or conversational phrasing into concise defect-description statements.
       Examples: "why does X not work?" → "X not working, failure in X"
                 "how come Y breaks?" → "Y breaking, Y error"
    4. EXPAND with closely related synonyms ONLY when they genuinely improve recall.
       Examples: "not loading" → add "blank screen, unresponsive, stuck"
                 "login issue" → add "sign-in failure, authentication error"
                 "slow" → add "performance issue, timeout, lag"
    5. If the query is already a clear defect description, keep it mostly as-is — do NOT over-expand.
    6. Do NOT hallucinate features, endpoints, or modules the user did not mention.
    7. Output ONLY the rephrased query — no explanation, no preamble, no quotes.

    User input:
    {user_query}

    Rephrased defect search query:
    """
)


@retry(max_attempts=3, initial_delay=2.0, backoff=2.0)
def _call_gemini_rephrase(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 300},
    )
    return response.text.strip()


def rephrase_query(user_query: str) -> str:
    """Rephrase natural-language input into a defect-search query.
    All original keywords and intent are preserved. Falls back to original on any failure."""
    if not user_query.strip():
        return user_query
    prompt = _REPHRASE_PROMPT.format(user_query=user_query)
    logger.info("Rephrasing query…")
    try:
        rephrased = _call_gemini_rephrase(prompt)
        logger.info("Rephrased: %s", rephrased)
        return rephrased or user_query
    except Exception as exc:
        logger.warning("Query rephrasing failed (%s) — using original.", exc)
        return user_query


# ── Embedding ─────────────────────────────────────────────────────────────────

@retry(max_attempts=4, initial_delay=2.0, backoff=2.0)
def embed_query(user_query: str) -> List[float]:
    genai = get_gemini_client()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=user_query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


# ── Pinecone Retrieval ────────────────────────────────────────────────────────

def retrieve_from_pinecone(
    query_vector: List[float],
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    index = get_pinecone_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    candidates = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        key = meta.get("key", match["id"])
        candidates.append(
            {
                "src":              meta.get("src", key),
                "key":              key,
                "reporter":         meta.get("reporter", ""),
                "status":           meta.get("status", ""),
                "summary":          meta.get("summary", ""),
                "description":      meta.get("description", ""),
                "comments":         meta.get("comments", ""),
                "similarity_score": round(float(match["score"]), 4),
            }
        )

    logger.info("Retrieved %d candidates from Pinecone.", len(candidates))
    return candidates


# ── Reranker ──────────────────────────────────────────────────────────────────

_RERANK_PROMPT = textwrap.dedent(
    """\
    You are a senior QA engineer at a SaaS CRM company specialising in defect triage and duplicate detection.
    Your job is to assess whether each existing ticket is a duplicate of a newly filed defect.

    NEW DEFECT:
    ═══════════════════════════════════════
    {user_query}
    ═══════════════════════════════════════

    Below are {n} existing tickets retrieved as potential duplicates.
    Score each on DUPLICATE LIKELIHOOD from 0 to 100 using these criteria:

    90–100  Definite duplicate — same root cause, same symptom, same failure point.
    70–89   Very likely duplicate — same core issue, minor differences in env/user/steps.
    50–69   Probable duplicate — same area of the product, overlapping symptoms.
    25–49   Related but distinct — same module or feature, different root cause.
    0–24    Not a duplicate — superficially similar but fundamentally different issue.

    Scoring rules:
    - Prioritise matching in this exact order: (1) feature/module name, (2) error symptom,
      (3) user action that triggers it, (4) environment or account details.
    - FEATURE AND MODULE NAMES ARE HARD SIGNALS — if the new defect names a specific feature,
      workflow, or module, any ticket about a distinctly different feature caps at 35, even if
      they share a keyword or broad product area. Do not conflate features just because they
      sound related or belong to the same category.
    - A different reporter or account does NOT lower the score if the defect itself matches.
    - Identical or near-identical summaries should score 90+.
    - A ticket with Status Done/Resolved that matches the defect is especially valuable — it may
      contain a known fix. This does NOT change the duplicate score, but note it in the reason.
    - Give a concise specific reason — mention the exact symptom, endpoint, or feature name.
    - Score every defect independently; do not inflate lower-ranked ones.

    Return ONLY valid JSON — a flat array ordered highest to lowest score.
    Each object must have exactly these three fields:
      "key"    : the defect key (string)
      "score"  : integer 0–100
      "reason" : one sentence referencing specific evidence

    Existing tickets:
    {defects_block}

    JSON:
    """
)


def _format_defects_for_rerank(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        status_line = f"   Status     : {c.get('status', '')}\n" if c.get("status") else ""
        desc = c["description"]
        cmts = c["comments"]
        lines.append(
            f"{i}. [{c['key']}]\n"
            f"   Summary    : {c['summary']}\n"
            f"{status_line}"
            f"   Description: {desc[:800]}{'…' if len(desc) > 800 else ''}\n"
            f"   Comments   : {cmts[:400]}{'…' if len(cmts) > 400 else ''}"
        )
    return "\n\n".join(lines)


@retry(max_attempts=4, initial_delay=3.0, backoff=2.0)
def _call_gemini_rerank(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 10000},
    )
    return response.text


def rerank_candidates(
    user_query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_RERANK,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    defects_block = _format_defects_for_rerank(candidates)
    prompt = _RERANK_PROMPT.format(
        user_query=user_query,
        n=len(candidates),
        defects_block=defects_block,
    )

    logger.info("Calling Gemini reranker (%s)…", RERANK_MODEL)
    raw = _call_gemini_rerank(prompt)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        ranked_list = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Reranker returned truncated JSON (%s) — falling back to similarity order.", exc
        )
        fallback = sorted(candidates, key=lambda c: c["similarity_score"], reverse=True)
        return [
            {
                "src":              c.get("src", c["key"]),
                "key":              c["key"],
                "reporter":         c.get("reporter", ""),
                "status":           c.get("status", ""),
                "summary":          c.get("summary", ""),
                "description":      c.get("description", ""),
                "comments":         c.get("comments", ""),
                "similarity_score": c["similarity_score"],
                "score":            0,
                "_rerank_reason":   "",
            }
            for c in fallback[:top_k]
        ]

    candidate_map = {c["key"]: c for c in candidates}

    merged: List[Dict[str, Any]] = []
    for item in ranked_list[:top_k]:
        key = item.get("key", "")
        base = candidate_map.get(key, {})
        merged.append(
            {
                "src":              base.get("src", key),
                "key":              key,
                "reporter":         base.get("reporter", ""),
                "status":           base.get("status", ""),
                "summary":          base.get("summary", ""),
                "description":      base.get("description", ""),
                "comments":         base.get("comments", ""),
                "similarity_score": base.get("similarity_score", 0.0),
                "score":            int(item.get("score", 0)),
                "_rerank_reason":   item.get("reason", ""),
            }
        )

    logger.info("Reranked to top %d results.", len(merged))
    return merged


# ── Per-Ticket Insight Generator ──────────────────────────────────────────────

_TICKET_INSIGHT_PROMPT = textwrap.dedent(
    """\
    You are a senior support analyst at a SaaS CRM company.
    Before writing a single word, read the ENTIRE ticket — Summary, Description, and every
    comment — carefully. All JSON string values must be plain text with no markdown, no bullet
    symbols, and no line breaks.

    ══════════════════════════════════════════════════
    SEARCH QUERY (what the user is investigating):
    {user_query}
    ══════════════════════════════════════════════════

    TICKET:
    Key         : {key}
    Status      : {status}
    Reporter    : {reporter}
    Summary     : {summary}
    Description :
    {description}

    Comments    :
    {comments}
    ══════════════════════════════════════════════════

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TASK 1 — ticket_summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Write 2–3 sentences synthesising THIS ticket from all its fields.

    • Name the exact feature or module affected and state what broke.
    • Describe the concrete observable behaviour — what the user saw or could not do.
    • Reference actual values (exact feature name, error message, or specific user action).
    • Synthesise — do NOT copy the Summary field verbatim.
    • If status is Closed/Resolved and comments document a fix → state it briefly.
      If closed but no fix is documented → say "Ticket was closed without a documented resolution."
      Never invent a resolution.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TASK 2 — insight  (Similarity + RCA if present)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Write 3–5 sentences as one flowing paragraph. Do NOT open with generic phrases like
    "This ticket directly relates to" or "This ticket is similar to the search query" —
    instead, lead with the specific shared symptom, feature, or technical finding.

    ── SIMILARITY (always required) ──────────────────
    State the precise symptom, failure pattern, or feature that overlaps with the search
    query. Be specific: name the exact module, action, error, or condition. If there are
    meaningful differences (different trigger, different user role, different severity) — note them.

    ── ROOT CAUSE (only for eligible statuses with documented evidence) ──────────────
    Status gate — only look for RCA if status is:
      Closed | Closed w/o Customer Ack | Resolved | Cancelled
    Current status: {status}
    If status is anything else → skip RCA entirely, write only the similarity.

    For eligible statuses, scan the FULL Description and ALL Comments for these RCA signals:

    ▸ Explicit cause statements:
      "root cause", "this happens because", "turned out to be", "found that", "the reason was",
      "the issue was", "this was caused by", "the problem is/was", "it was happening because",
      "this occurred because", "after investigation", "upon investigation", "we identified",
      "we found that", "discovered that", "investigation shows", "this is because"

    ▸ Technical fix or resolution WITH specifics:
      "fixed by", "resolved by", "the fix was", "we deployed", "we updated", "we changed",
      "we rolled back", "patch applied", "we corrected", "the change was", "fix has been deployed",
      "we have fixed this by" — ONLY when followed by a specific technical description of WHAT changed.
      A bare "Issue has been fixed" or "Closing ticket" with nothing after it does NOT qualify.

    ▸ Hard technical evidence in the ticket:
      Specific error codes, exception names, stack traces, API responses, database values,
      HTTP status codes with context, payload field names or data conditions named explicitly.

    ▸ Engineering investigation findings:
      A specific component, code path, config key, database table, API endpoint, or
      third-party service identified as the failure point.

    ▸ Technical explanation of WHY it happened:
      Race condition, missing validation, wrong field mapping, NULL/empty value handling,
      data inconsistency, cache issue, wrong environment config, third-party rate limit,
      permission gap, duplicate record collision, data migration error — any named root cause.

    DECISION:
    → Found at least one signal above with actual specific detail → include the root cause
      in the insight paragraph. Quote or closely paraphrase the evidence from the ticket.
    → Found only generic closure notes ("Fixed", "Duplicate", "Closing") with no technical
      detail → do NOT include root cause; write only the similarity.
    → No signals found anywhere → write only the similarity.

    When including root cause: weave it naturally into the paragraph after the similarity.
    State the specific cause, technical detail, or fix — directly traceable to ticket text.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ACCURACY CONTRACT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Every factual claim must be directly traceable to text in Description or Comments.
    • Never invent error codes, fix steps, component names, or causes absent from the ticket.
    • Never infer a fix or resolution from ticket status alone.
    • Never use generic filler — every sentence must reference a specific detail from the data.

    Return ONLY valid JSON — no code fences, nothing before or after:
    {{"ticket_summary": "...", "insight": "..."}}
    """
)


@retry(max_attempts=4, initial_delay=3.0, backoff=2.0)
def _call_gemini_insight(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 1500},
    )
    return response.text.strip()


_RCA_ELIGIBLE_STATUSES = frozenset({
    "closed",
    "closed w/o customer ack",
    "resolved",
    "canceled",
    "cancelled",
})


def _is_rca_eligible(status: str) -> bool:
    """Return True only for statuses that can contain a documented root cause."""
    return (status or "").strip().lower() in _RCA_ELIGIBLE_STATUSES



def generate_ticket_insights(
    user_query: str,
    reranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate AI ticket summary + RCA/similarity insight for each reranked result."""
    enriched = []
    for i, item in enumerate(reranked, 1):
        logger.info("Generating insight %d/%d (%s)…", i, len(reranked), item["key"])
        try:
            prompt = _TICKET_INSIGHT_PROMPT.format(
                user_query=user_query,
                key=item["key"],
                status=item.get("status", "Unknown"),
                reporter=item.get("reporter", "Unknown"),
                summary=item["summary"],
                description=item["description"],
                comments=item["comments"],
            )
            raw = _call_gemini_insight(prompt)

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()

            data = json.loads(cleaned)
            ticket_summary = str(data.get("ticket_summary") or item["summary"]).strip()
            insight        = str(data.get("insight") or "").strip()

        except json.JSONDecodeError:
            logger.warning("Insight JSON parse failed for %s — using fallback.", item["key"])
            ticket_summary = item["summary"]
            insight        = ""
        except Exception as exc:
            logger.warning("Insight generation failed for %s (%s) — skipping.", item["key"], exc)
            ticket_summary = item["summary"]
            insight        = ""

        enriched.append({
            **item,
            "ticket_summary": ticket_summary,
            "insight":        insight,
        })

    return enriched


# ── Centralized Summary ───────────────────────────────────────────────────────

_CENTRALIZED_SUMMARY_PROMPT = textwrap.dedent(
    """\
    You are a senior support intelligence analyst at a SaaS CRM company.

    A user searched for:
    "{original_query}"

    The system found {n} similar tickets in the knowledge base. Read all of them carefully.

    {tickets_block}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    YOUR TASK
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Write a 2–3 sentence centralized summary that directly answers what the user was looking for.

    Rules:
    • Lead directly with the key finding — DO NOT start with "Yes, there are..." or "The system found..."
      or any preamble. Open with the substance: the pattern, the root cause, or the core insight.
    • Identify patterns across tickets: shared root cause, same feature failing, same error type,
      recurring symptom. If all tickets converge on one cause, state it clearly.
    • Include resolution context when meaningful — e.g. how many of the {n} tickets are resolved
      and, if the root cause was identified, what the fix was.
    • Be specific — reference actual feature names, root causes, or fix details from the tickets.
    • If tickets vary widely with no clear pattern, describe the range of issues found.
    • Write as a knowledgeable colleague giving a concise briefing, not a bot listing data points.
    • Plain text only — no markdown, no bullet points, no headers.

    Summary:
    """
)


def _format_tickets_for_summary(enriched: List[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(enriched, 1):
        desc_snip = item.get("description", "")[:400]
        if len(item.get("description", "")) > 400:
            desc_snip += "…"
        lines.append(
            f"TICKET {i}: [{item['key']}] — Status: {item.get('status', 'Unknown')}\n"
            f"Summary    : {item['summary']}\n"
            f"Description: {desc_snip}\n"
            f"Analysis   : {item.get('insight', '')}"
        )
    return "\n\n---\n\n".join(lines)


@retry(max_attempts=3, initial_delay=3.0, backoff=2.0)
def _call_gemini_centralized(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.15, "max_output_tokens": 512},
    )
    return response.text.strip()


def generate_centralized_summary(
    original_query: str,
    enriched: List[Dict[str, Any]],
) -> str:
    """Generate a single intelligence summary across all retrieved tickets."""
    if not enriched:
        return ""
    tickets_block = _format_tickets_for_summary(enriched)
    prompt = _CENTRALIZED_SUMMARY_PROMPT.format(
        original_query=original_query,
        n=len(enriched),
        tickets_block=tickets_block,
    )
    logger.info("Generating centralized summary…")
    try:
        return _call_gemini_centralized(prompt)
    except Exception as exc:
        logger.warning("Centralized summary failed (%s).", exc)
        return ""


# ── Output Builder ────────────────────────────────────────────────────────────

def build_final_output(
    user_query: str,
    search_query: str,
    enriched_results: List[Dict[str, Any]],
    centralized_summary: str,
) -> Dict[str, Any]:
    results = []
    for item in enriched_results:
        results.append(
            {
                "src":              item.get("src", item["key"]),
                "key":              item["key"],
                "reporter":         item.get("reporter", ""),
                "status":           item.get("status", ""),
                "summary":          item["summary"],
                "score":            item["score"],
                "similarity_score": item["similarity_score"],
                "ticket_summary":   item.get("ticket_summary", ""),
                "insight":          item.get("insight", ""),
            }
        )
    return {
        "query":               user_query,
        "search_query":        search_query,
        "centralized_summary": centralized_summary,
        "results":             results,
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def find_duplicates(
    user_query: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
) -> Dict[str, Any]:
    logger.info("=== Duplicate detection pipeline started ===")

    # Step 1: Rephrase for retrieval — preserves all keywords, adds synonyms
    search_query = rephrase_query(user_query)

    # Step 2: Embed rephrased query and retrieve candidates
    query_vector = embed_query(search_query)
    candidates = retrieve_from_pinecone(query_vector, top_k=top_k_retrieval)

    if not candidates:
        logger.warning("No candidates retrieved from Pinecone.")
        return {
            "query":               user_query,
            "search_query":        search_query,
            "centralized_summary": "",
            "results":             [],
        }

    # Step 3: Rerank using rephrased query (more keyword-precise signal)
    reranked = rerank_candidates(search_query, candidates, top_k=top_k_rerank)

    # Step 4: Per-ticket insights using original query (preserves user intent for AI)
    enriched = generate_ticket_insights(user_query, reranked)

    # Step 5: Centralized intelligence summary across all tickets
    centralized_summary = generate_centralized_summary(user_query, enriched)

    output = build_final_output(user_query, search_query, enriched, centralized_summary)

    logger.info("=== Pipeline complete. Returning %d results. ===", len(output["results"]))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find duplicate defects for a given query."
    )
    parser.add_argument("--query", required=True, help="New defect description.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RERANK,
        help=f"Number of final results to return (default: {TOP_K_RERANK}).",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help=f"Candidates to pull from Pinecone before reranking (default: {TOP_K_RETRIEVAL}).",
    )
    args = parser.parse_args()

    result = find_duplicates(
        user_query=args.query,
        top_k_retrieval=args.retrieval_k,
        top_k_rerank=args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
