import argparse
import json
import re
import textwrap
from typing import Any, Dict, List, Tuple

from utils import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    TOP_K_QUERY_RETRIEVAL,
    TOP_K_QUERY_RERANK_POOL,
    MIN_QUERY_RELEVANCE_SCORE,
    QUERIES_EMBEDDING_DIMENSION,
    logger,
    retry,
    normalize_l2,
    get_gemini_client,
    get_pinecone_index,
    get_pinecone_queries_index,
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
def embed_query(
    user_query: str,
    output_dimensionality: int | None = None,
) -> List[float]:
    genai = get_gemini_client()
    kwargs: Dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "content": user_query,
        "task_type": "RETRIEVAL_QUERY",
    }
    if output_dimensionality is not None:
        kwargs["output_dimensionality"] = output_dimensionality

    result = genai.embed_content(**kwargs)
    embedding = result["embedding"]
    if output_dimensionality is not None and output_dimensionality != 3072:
        embedding = normalize_l2(embedding)
    return embedding


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


# ── Query Mode (queries-data index) ───────────────────────────────────────────

TOP_K_QUERY_SOURCES = 5

_QUERY_REPHRASE_PROMPT = textwrap.dedent(
    """\
    You are a senior CSE knowledge analyst at a SaaS CRM company (Recruit CRM).
    A support agent is asking a question. Your job is to produce ONE optimised search query
    that will retrieve past CSE tickets whose resolutions actually answer this question.

    Think step-by-step (internally — do not output your reasoning):
    1. What is the user really asking? (cause, how-to, policy, billing, configuration, etc.)
    2. What specific product area, feature, module, or workflow is involved?
    3. What answer signals would appear in a helpful past ticket? (resolution in comments,
       explanation from engineer, confirmed fix, workaround, root cause)

    STRICT RULES:
    1. PRESERVE every specific feature name, module, product name, error message, account detail,
       pricing term, integration name, and technical keyword — never substitute or drop them.
    2. Frame the query as a QUESTION or investigation that matches how past tickets were written,
       not as a defect report. Example: "Why are call charges $5 instead of $0.40 for 10-min calls?"
       not "Call billing charge discrepancy failure".
    3. Include the core intent words: why, how, what, check, verify, explain, cause, resolve, etc.
    4. Add 2–4 closely related terms ONLY if they improve recall for the same question
       (e.g. "call cost" → add "call rate, per-minute charge, billing").
    5. Do NOT broaden to unrelated product areas. Do NOT hallucinate details.
    6. Output ONLY the rephrased search query — one line, no quotes, no explanation.

    User question:
    {user_query}

    Optimised search query:
    """
)


def rephrase_query_question(user_query: str) -> str:
    """Rephrase a natural-language question for queries-data retrieval."""
    if not user_query.strip():
        return user_query
    prompt = _QUERY_REPHRASE_PROMPT.format(user_query=user_query)
    logger.info("Rephrasing query-mode question…")
    try:
        rephrased = _call_gemini_rephrase(prompt)
        logger.info("Rephrased: %s", rephrased)
        return rephrased or user_query
    except Exception as exc:
        logger.warning("Query rephrasing failed (%s) — using original.", exc)
        return user_query


def retrieve_from_queries_index(
    query_vector: List[float],
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    index = get_pinecone_queries_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    candidates = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        key = meta.get("key", match["id"])
        text = meta.get("text", "")
        if not text:
            text = "\n".join(
                part for part in (
                    f"Summary: {meta.get('summary', '')}" if meta.get("summary") else "",
                    f"Description: {meta.get('description', '')}" if meta.get("description") else "",
                    f"Comments: {meta.get('comments', '')}" if meta.get("comments") else "",
                ) if part
            )
        candidates.append(
            {
                "src":              meta.get("src", key),
                "key":              key,
                "reporter":         meta.get("reporter", ""),
                "status":           meta.get("status", ""),
                "summary":          meta.get("summary", ""),
                "priority":         meta.get("priority", ""),
                "team":             meta.get("team", ""),
                "text":             text,
                "similarity_score": round(float(match["score"]), 4),
            }
        )

    logger.info("Retrieved %d candidates from queries-data index.", len(candidates))
    return candidates


# ── Keyword Hybrid Retrieval (queries-data) ───────────────────────────────────
# Index vectors were embedded with a different pipeline than query embeddings,
# so pure vector search can miss obviously relevant tickets. Keyword retrieval
# supplements vector results by matching feature names and terms in ticket text.

_QUERIES_METADATA_CACHE: Dict[str, Dict[str, Any]] | None = None

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "with", "from", "into", "about",
    "for", "on", "in", "to", "of", "and", "or", "not", "no", "it", "its",
    "this", "that", "these", "those", "at", "by", "as", "if", "when",
    "where", "how", "what", "why", "who", "which", "there", "their",
    "they", "them", "we", "our", "you", "your", "i", "my", "me", "he",
    "she", "his", "her", "but", "so", "than", "too", "very", "just",
    "also", "any", "all", "some", "most", "other", "such", "only",
    "same", "both", "each", "few", "more", "most", "own", "same",
    "experiencing", "issue", "issues", "problem", "problems", "please",
    "check", "help", "need", "want", "know", "tell", "show", "see",
})


def _load_queries_metadata_cache() -> Dict[str, Dict[str, Any]]:
    """Load all ticket metadata from queries-data index (cached after first call)."""
    global _QUERIES_METADATA_CACHE
    if _QUERIES_METADATA_CACHE is not None:
        return _QUERIES_METADATA_CACHE

    index = get_pinecone_queries_index()
    all_ids: List[str] = []
    pagination_token = None
    while True:
        kwargs: Dict[str, Any] = {"limit": 100}
        if pagination_token:
            kwargs["pagination_token"] = pagination_token
        page = index.list_paginated(**kwargs)
        for item in page.vectors:
            all_ids.append(item.id)
        if not page.pagination or not page.pagination.next:
            break
        pagination_token = page.pagination.next

    logger.info("Loading queries-data metadata cache (%d tickets)…", len(all_ids))
    cache: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(all_ids), 100):
        batch = all_ids[i : i + 100]
        fetched = index.fetch(ids=batch)
        for vid, data in (fetched.vectors or {}).items():
            meta = data.metadata or {}
            text = meta.get("text", "")
            if not text:
                text = "\n".join(
                    part for part in (
                        f"Summary: {meta.get('summary', '')}" if meta.get("summary") else "",
                        f"Description: {meta.get('description', '')}" if meta.get("description") else "",
                        f"Comments: {meta.get('comments', '')}" if meta.get("comments") else "",
                    ) if part
                )
            cache[vid] = {
                "src":      meta.get("src", vid),
                "key":      meta.get("key", vid),
                "reporter": meta.get("reporter", ""),
                "status":   meta.get("status", ""),
                "summary":  meta.get("summary", ""),
                "priority": meta.get("priority", ""),
                "team":     meta.get("team", ""),
                "text":     text,
            }

    _QUERIES_METADATA_CACHE = cache
    logger.info("Queries metadata cache ready (%d records).", len(cache))
    return cache


def _extract_search_terms(query: str) -> Tuple[List[str], List[str]]:
    """Return (multi_word_phrases, single_terms) extracted from the query."""
    phrases: List[str] = []
    # Quoted phrases: "Last Communication"
    for m in re.finditer(r'["\']([^"\']{3,60})["\']', query):
        phrases.append(m.group(1).strip().lower())
    # Capitalised multi-word phrases: Last Communication, Recruit CRM
    for m in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', query):
        phrases.append(m.group(1).strip().lower())
    # Also try title-casing significant lowercase multi-word from query
    words_raw = re.findall(r'[a-zA-Z]{3,}', query)
    words = [w.lower() for w in words_raw if w.lower() not in _STOPWORDS]

    # Build 2-word phrases from consecutive significant words
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i+1]}"
        if pair not in phrases:
            phrases.append(pair)

    terms = list(dict.fromkeys(w for w in words if len(w) >= 4))
    phrases = list(dict.fromkeys(phrases))
    return phrases, terms


def _keyword_overlap_score(haystack: str, phrases: List[str], terms: List[str]) -> float:
    """Score how well haystack matches query phrases and terms (0.0–1.0)."""
    if not haystack:
        return 0.0
    text = haystack.lower()
    score = 0.0
    max_score = len(phrases) * 3.0 + len(terms) * 1.0
    if max_score == 0:
        return 0.0
    for phrase in phrases:
        if phrase in text:
            score += 3.0
        else:
            # Partial: all words of phrase present
            pwords = phrase.split()
            if len(pwords) > 1 and all(w in text for w in pwords):
                score += 2.0
    for term in terms:
        if term in text:
            score += 1.0
        elif len(term) >= 6:
            stem = term[: max(5, len(term) - 3)]
            if stem in text:
                score += 0.5
    return min(score / max_score, 1.0)


def keyword_retrieve_from_queries_index(
    user_query: str,
    top_k: int = 15,
    min_score: float = 0.15,
) -> List[Dict[str, Any]]:
    """Retrieve tickets by keyword/phrase overlap in summary and text."""
    cache = _load_queries_metadata_cache()
    phrases, terms = _extract_search_terms(user_query)
    if not phrases and not terms:
        return []

    logger.info("Keyword search — phrases: %s, terms: %s", phrases[:5], terms[:8])
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for key, record in cache.items():
        haystack = f"{record.get('summary', '')} {record.get('text', '')}"
        kw_score = _keyword_overlap_score(haystack, phrases, terms)
        if kw_score >= min_score:
            scored.append((kw_score, key, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = []
    for kw_score, key, record in scored[:top_k]:
        candidates.append({
            **record,
            "key":              key,
            "similarity_score": 0.0,
            "keyword_score":    round(kw_score, 4),
        })

    logger.info("Keyword retrieval found %d candidates (top score %.3f).",
                len(candidates), scored[0][0] if scored else 0.0)
    return candidates


def _merge_query_candidates(
    *candidate_lists: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge retrieval results from multiple sources, combining vector + keyword scores."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for candidates in candidate_lists:
        for c in candidates:
            key = c["key"]
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(c)
            else:
                existing["similarity_score"] = max(
                    existing.get("similarity_score", 0.0),
                    c.get("similarity_score", 0.0),
                )
                existing["keyword_score"] = max(
                    existing.get("keyword_score", 0.0),
                    c.get("keyword_score", 0.0),
                )

    def _combined_score(c: Dict[str, Any]) -> float:
        vec = c.get("similarity_score", 0.0)
        kw  = c.get("keyword_score", 0.0)
        # Keyword matches on exact feature names are strong signals — weight equally
        return max(vec, kw * 0.85) + min(vec, kw) * 0.15

    merged = sorted(by_key.values(), key=_combined_score, reverse=True)
    logger.info("Merged to %d unique query candidates.", len(merged))
    return merged


def retrieve_query_candidates(
    user_query: str,
    search_query: str,
    top_k: int = TOP_K_QUERY_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """Retrieve using vector search (dual-query) + keyword hybrid search."""
    per_query_k = max(top_k // 2 + 5, 15)
    vec_original = embed_query(user_query, output_dimensionality=QUERIES_EMBEDDING_DIMENSION)
    vec_search = embed_query(search_query, output_dimensionality=QUERIES_EMBEDDING_DIMENSION)
    from_original = retrieve_from_queries_index(vec_original, top_k=per_query_k)
    from_search   = retrieve_from_queries_index(vec_search, top_k=per_query_k)

    # Keyword pass on both original and rephrased query for maximum recall
    from_kw_original = keyword_retrieve_from_queries_index(user_query, top_k=15)
    from_kw_search   = keyword_retrieve_from_queries_index(search_query, top_k=10)

    merged = _merge_query_candidates(
        from_original, from_search, from_kw_original, from_kw_search
    )
    return merged[:top_k]


_QUERY_RERANK_PROMPT = textwrap.dedent(
    """\
    You are a senior CSE analyst at a SaaS CRM company (Recruit CRM).
    A colleague asked a question. Below are past CSE query tickets retrieved by vector search.
    Your job is NOT to find "similar" tickets — it is to find tickets that contain information
    that would help you give a CORRECT, ACTIONABLE answer to the colleague's question.

    COLLEAGUE'S QUESTION:
    ═══════════════════════════════════════
    {user_query}
    ═══════════════════════════════════════

    Score each ticket on ANSWER USEFULNESS from 0 to 100:

    90–100  Contains a direct answer, confirmed resolution, or clear explanation that addresses
            this exact question. Comments or description state what happened and why/how to fix.
    70–89   Strong partial answer — same scenario with useful resolution steps or root cause,
            even if not identical in every detail.
    50–69   Informative context — related question with some transferable insight, but not a
            full answer to this specific question.
    25–49   Same product area or shared keywords, but answers a DIFFERENT question. Not useful.
    0–24    Irrelevant — does not help answer the colleague's question at all.

    CRITICAL SCORING RULES:
    - Keyword overlap alone is NOT enough. A ticket about "LinkedIn integration" must score low
      for a question about "call billing rates" even if both mention "customer account".
    - Read the FULL ticket content, especially Comments — answers often appear only in the
      comment thread (engineer replies, confirmed fixes, "this was because...", workarounds).
    - Prioritise tickets where someone explicitly answered the question or closed with a reason.
    - Closed/Resolved tickets with documented outcomes in comments score higher.
    - If the ticket asks the same question but was never answered, score 30–45 max.
    - Be strict: most retrieved tickets should score below 50 if they don't actually help.

    Return ONLY valid JSON — a flat array ordered highest to lowest score.
    Each object must have exactly these four fields:
      "key"        : the ticket key (string)
      "score"      : integer 0–100
      "answerable" : true if score >= 50, else false
      "reason"     : one sentence — what specific evidence helps (or doesn't help) answer the question

    Historical tickets:
    {tickets_block}

    JSON:
    """
)


def _format_queries_for_rerank(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c.get("text", "")
        status_line = f"   Status  : {c.get('status', '')}\n" if c.get("status") else ""
        team_line = f"   Team    : {c.get('team', '')}\n" if c.get("team") else ""
        lines.append(
            f"{i}. [{c['key']}]\n"
            f"   Summary : {c.get('summary', '')}\n"
            f"{status_line}"
            f"{team_line}"
            f"   Content : {text[:2500]}{'…' if len(text) > 2500 else ''}"
        )
    return "\n\n".join(lines)


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def rerank_query_candidates(
    user_query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_QUERY_SOURCES,
    rerank_pool: int = TOP_K_QUERY_RERANK_POOL,
    min_score: int = MIN_QUERY_RELEVANCE_SCORE,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    pool = candidates[:rerank_pool]
    tickets_block = _format_queries_for_rerank(pool)
    prompt = _QUERY_RERANK_PROMPT.format(
        user_query=user_query,
        tickets_block=tickets_block,
    )

    logger.info("Calling Gemini query reranker (%s) on %d candidates…", RERANK_MODEL, len(pool))
    raw = _call_gemini_rerank(prompt)

    try:
        ranked_list = _parse_json_array(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Query reranker returned truncated JSON (%s) — falling back to similarity order.", exc
        )
        fallback = sorted(candidates, key=lambda c: c["similarity_score"], reverse=True)
        return [
            {**c, "score": 0, "answerable": False, "_rerank_reason": ""}
            for c in fallback[:top_k]
        ]

    candidate_map = {c["key"]: c for c in pool}

    merged: List[Dict[str, Any]] = []
    for item in ranked_list:
        key = item.get("key", "")
        base = candidate_map.get(key)
        if not base:
            continue
        score = int(float(item.get("score", 0)))
        answerable = bool(item.get("answerable", score >= min_score))
        if score < min_score:
            answerable = False
        merged.append(
            {
                **base,
                "key":            key,
                "score":          score,
                "answerable":     answerable,
                "_rerank_reason": item.get("reason", ""),
            }
        )

    answerable = [m for m in merged if m.get("answerable")]
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []

    for m in answerable:
        if len(result) >= top_k:
            break
        result.append(m)
        seen.add(m["key"])

    for m in merged:
        if len(result) >= top_k:
            break
        if m["key"] not in seen:
            result.append(m)
            seen.add(m["key"])

    for c in pool:
        if len(result) >= top_k:
            break
        if c["key"] not in seen:
            result.append(
                {**c, "score": 0, "answerable": False, "_rerank_reason": ""}
            )
            seen.add(c["key"])

    if not answerable:
        logger.warning(
            "No tickets scored >= %d — returning top %d by rerank order.", min_score, top_k
        )

    logger.info(
        "Query reranked: %d answerable of %d scored; returning %d sources.",
        len(answerable), len(merged), len(result),
    )
    return result


# ── Per-ticket Query Insight ──────────────────────────────────────────────────

_QUERY_TICKET_INSIGHT_PROMPT = textwrap.dedent(
    """\
    You are a senior CSE analyst at Recruit CRM. Read the ticket below completely — every part
    of the text including summary, description, and comments.
    All JSON string values must be plain text with no markdown, no bullet symbols, no line breaks.

    ══════════════════════════════════════════════════
    USER'S QUESTION:
    {user_query}
    ══════════════════════════════════════════════════

    TICKET:
    Key    : {key}
    Status : {status}
    Team   : {team}
    Summary: {summary}

    Full content (summary + description + comments):
    {text}
    ══════════════════════════════════════════════════

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TASK 1 — ticket_summary (2–3 sentences)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Synthesise what happened in this ticket: what was the customer's question or issue, what
    the CSE team investigated or found, and what was the outcome (if any).
    • Reference the actual feature, module, or error involved — do not copy the Summary verbatim.
    • If the ticket was closed with a resolution documented in comments, include it briefly.
    • If closed without a documented resolution, say "Closed without a documented resolution."
    • Never invent an outcome not stated in the ticket.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TASK 2 — relevance (1–2 sentences)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Explain specifically how this ticket is relevant to the user's question.
    • State the exact shared element: same feature, same symptom, same error, same type of
      confusion, or same resolution that could apply.
    • If the ticket only partially overlaps, say what it covers and what it doesn't.
    • Do NOT open with "This ticket is relevant because" — lead with the shared detail itself.
    • If this ticket is not relevant to the question at all, write "Not directly relevant to the question."

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ACCURACY CONTRACT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Every claim must trace directly to the ticket text. Never invent.
    • No markdown, no bullets, no line breaks in any JSON value.

    Return ONLY valid JSON — no code fences, nothing before or after:
    {{"ticket_summary": "...", "relevance": "..."}}
    """
)


@retry(max_attempts=4, initial_delay=3.0, backoff=2.0)
def _call_gemini_query_insight(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 800},
    )
    return response.text.strip()


def generate_query_ticket_insights(
    user_query: str,
    ranked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate per-ticket AI summary + relevance insight for each ranked query source."""
    enriched = []
    for i, item in enumerate(ranked, 1):
        logger.info("Generating query insight %d/%d (%s)…", i, len(ranked), item["key"])
        text = item.get("text", item.get("summary", ""))
        try:
            prompt = _QUERY_TICKET_INSIGHT_PROMPT.format(
                user_query=user_query,
                key=item["key"],
                status=item.get("status", "Unknown"),
                team=item.get("team", ""),
                summary=item.get("summary", ""),
                text=text[:4000],
            )
            raw = _call_gemini_query_insight(prompt)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            data = json.loads(cleaned)
            ticket_summary = str(data.get("ticket_summary") or item.get("summary", "")).strip()
            relevance      = str(data.get("relevance") or "").strip()
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Query insight failed for %s (%s) — using fallback.", item["key"], exc)
            ticket_summary = item.get("summary", "")
            relevance      = ""
        enriched.append({**item, "ticket_summary": ticket_summary, "relevance": relevance})
    return enriched


# ── Query-mode Answer Synthesis ───────────────────────────────────────────────

_QUERY_ANSWER_PROMPT = textwrap.dedent(
    """\
    You are a senior CSE analyst at Recruit CRM. A colleague asked a question and you have
    reviewed the most relevant past CSE query tickets in full detail.
    Your job is to write a thoughtful, accurate answer grounded ONLY in the ticket content below.

    COLLEAGUE'S QUESTION:
    "{user_query}"

    RELEVANT PAST TICKETS (full content):
    {tickets_block}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    HOW TO WRITE THE ANSWER
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Write 3–5 paragraphs in plain text (no markdown, bullets, headers, or ticket keys).

    TONE:
    • Use hedged, evidence-based language — this is guidance from past cases, not a final verdict.
    • For confirmed resolutions documented in ticket comments, you may say "In a similar past case,
      the issue was resolved by...". For probable causes without hard evidence, use "One possible
      reason could be...", "This might be related to...", "It's worth checking whether...".
    • If multiple tickets point to the same cause, you can say that pattern was seen across cases.
    • If tickets conflict or give incomplete answers, say so honestly.

    STRUCTURE:
    Paragraph 1 — Lead with the most likely or best-supported explanation. Reference what the
    past tickets suggest, using appropriately hedged language.

    Paragraph 2 — Supporting detail: specific error messages, conditions, account types,
    configurations, or workflows mentioned in the tickets that are relevant.

    Paragraph 3 — Additional possible causes or angles if the evidence suggests multiple reasons.

    Paragraph 4 — Recommended next steps for the colleague to try or verify, based on what
    past tickets documented as useful investigation steps or resolutions.

    Paragraph 5 (optional) — Honest caveat: if evidence is partial, say what is confirmed vs.
    still needs investigation. If no ticket directly answered the question, say so clearly.

    STRICT RULES:
    • Every factual claim must be directly traceable to the ticket text provided. Never invent.
    • Do NOT mention ticket keys, "according to ticket X", or "the knowledge base" in the answer.
    • Do NOT use bullet points, headers, or markdown.
    • Sound like a knowledgeable colleague sharing informed possibilities, not a search result.

    Answer:
    """
)


def _format_tickets_for_query_answer(sources: List[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(sources, 1):
        text = item.get("text", "")
        lines.append(
            f"TICKET {i}: [{item['key']}] — Status: {item.get('status', 'Unknown')}\n"
            f"Summary: {item.get('summary', '')}\n"
            f"Team: {item.get('team', '')}\n"
            f"Full content:\n{text[:3500]}{'…' if len(text) > 3500 else ''}"
        )
    return "\n\n---\n\n".join(lines)


@retry(max_attempts=3, initial_delay=3.0, backoff=2.0)
def _call_gemini_query_answer(prompt: str) -> str:
    genai = get_gemini_client()
    model = genai.GenerativeModel(RERANK_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.15, "max_output_tokens": 2200},
    )
    return response.text.strip()


def generate_query_answer(
    user_query: str,
    enriched: List[Dict[str, Any]],
) -> str:
    """Synthesise a hedged answer grounded in the full enriched ticket data."""
    if not enriched:
        return ""
    tickets_block = _format_tickets_for_query_answer(enriched)
    prompt = _QUERY_ANSWER_PROMPT.format(
        user_query=user_query,
        tickets_block=tickets_block,
    )
    logger.info("Generating query-mode answer from enriched ticket data…")
    try:
        return _call_gemini_query_answer(prompt)
    except Exception as exc:
        logger.warning("Query answer generation failed (%s).", exc)
        return ""


def find_query_answer(
    user_query: str,
    top_k_retrieval: int = TOP_K_QUERY_RETRIEVAL,
    top_k_sources: int = TOP_K_QUERY_SOURCES,
) -> Dict[str, Any]:
    """Answer a user question using the queries-data Pinecone index."""
    logger.info("=== Query-answer pipeline started ===")

    search_query = rephrase_query_question(user_query)
    candidates = retrieve_query_candidates(
        user_query, search_query, top_k=top_k_retrieval
    )

    if not candidates:
        logger.warning("No candidates retrieved from queries-data index.")
        return {
            "query":        user_query,
            "search_query": search_query,
            "answer":       "",
            "sources":      [],
        }

    ranked = rerank_query_candidates(user_query, candidates, top_k=top_k_sources)

    # Generate per-ticket AI summaries + relevance insights
    enriched = generate_query_ticket_insights(user_query, ranked)

    # Synthesise the top-level answer from full ticket data
    answer = generate_query_answer(user_query, enriched)

    sources = [
        {
            "src":              item.get("src", item["key"]),
            "key":              item["key"],
            "reporter":         item.get("reporter", ""),
            "status":           item.get("status", ""),
            "summary":          item.get("summary", ""),
            "priority":         item.get("priority", ""),
            "team":             item.get("team", ""),
            "score":            item.get("score", 0),
            "similarity_score": item.get("similarity_score", 0.0),
            "ticket_summary":   item.get("ticket_summary", ""),
            "relevance":        item.get("relevance", ""),
        }
        for item in enriched
    ]

    logger.info("=== Query-answer pipeline complete. %d sources cited. ===", len(sources))
    return {
        "query":        user_query,
        "search_query": search_query,
        "answer":       answer,
        "sources":      sources,
    }


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
