import html as _html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from query import find_duplicates, find_query_answer
import jira_cloud_auth as jca
from fetch_jira_jql import iter_all_issues, normalize_issue, SEARCH_FIELDS

JIRA_BASE   = "https://rcrm.atlassian.net/servicedesk/customer/portal/2/"
JIRA_BROWSE = "https://rcrm.atlassian.net/browse/"

RESOLVED_STATUSES = {
    "done", "closed", "resolved", "complete", "fixed",
    "cancelled", "canceled", "rejected", "duplicate",
    "won't fix", "wont fix", "cannot reproduce",
}

# Chart palette — sky blue spectrum
CHART_PRIMARY   = "#0EA5E9"
CHART_SECONDARY = "#38BDF8"
CHART_ACCENT    = "#0284C7"

STATUS_COLORS = {
    "Done":                      "#10B981",
    "Cancelled":                 "#94A3B8",
    "Rejected":                  "#CBD5E1",
    "To Do":                     "#F43F5E",
    "Investigating":             "#F59E0B",
    "Development In Progress":   "#F59E0B",
    "Reopened":                  "#F43F5E",
    "Waiting for customer":      "#06B6D4",
    "Waiting For CSA":           "#06B6D4",
    "Waiting For Third Party":   "#7DD3FC",
    "Waiting For Engineer":      "#7DD3FC",
    "Waiting for Product Team":  "#7DD3FC",
}

PRIORITY_COLORS = {
    "Blocker":  "#991B1B",
    "Critical": "#F43F5E",
    "High":     "#F59E0B",
    "Medium":   "#0EA5E9",
    "Low":      "#10B981",
    "Lowest":   "#6EE7B7",
    "None":     "#CBD5E1",
}

PROJECT_COLORS = [
    "#0EA5E9", "#38BDF8", "#0284C7", "#10B981", "#F59E0B",
    "#F43F5E", "#7DD3FC", "#34D399", "#0369A1", "#22D3EE", "#FBBF24",
]

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#475569"),
    margin=dict(l=0, r=0, t=10, b=0),
)

st.set_page_config(
    page_title="CSE Ticket Intelligence",
    page_icon="🎯",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary:       #0284C7;
        --primary-dark:  #0369A1;
        --accent:        #0EA5E9;
        --accent-light:  #E0F2FE;
        --surface:       #FFFFFF;
        --surface-muted: #FFFFFF;
        --text-primary:  #0F172A;
        --text-secondary:#64748B;
        --text-muted:    #94A3B8;
        --border:        #E2E8F0;
        --shadow-sm:     0 1px 3px rgba(15,23,42,0.06);
        --shadow-md:     0 4px 12px rgba(14,165,233,0.10);
        --shadow-lg:     0 8px 24px rgba(14,165,233,0.12);
        --radius:        12px;
        --radius-lg:     16px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* ── App shell — clean white + light blue ── */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #F0F9FF 50%, #F8FAFC 100%);
    }
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    header[data-testid="stHeader"] { height: 0 !important; }

    /* ── Tabs — clean pill navigation ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #F1F5F9;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px;
        box-shadow: none;
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px !important;
        padding: 0 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary) !important;
        background: #E0F2FE !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary) !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(2,132,199,0.25);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── App header (above tabs) ── */
    .app-header {
        margin-bottom: 0.25rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 4px 0;
        color: #0F172A;
    }
    .app-subtitle {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }
    .mode-hint {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }
    .page-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0 0 6px 0;
        color: #0F172A;
        background: none;
        -webkit-text-fill-color: #0F172A;
    }
    .page-header {
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
    }
    .page-subtitle {
        font-size: 14px;
        color: var(--text-secondary);
        margin: 0;
    }

    /* ── Cards ── */
    .result-card, .query-result-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .result-card:hover, .query-result-card:hover {
        box-shadow: var(--shadow-md);
        border-color: #BAE6FD;
    }
    .card-meta { display:flex; align-items:center; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
    .ticket-chip {
        background: var(--primary);
        color: #fff !important;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 8px;
        text-decoration: none !important;
        letter-spacing: 0.03em;
        box-shadow: 0 2px 6px rgba(2,132,199,0.25);
        transition: background 0.15s ease;
    }
    .ticket-chip:hover {
        background: var(--primary-dark);
    }
    .reporter-chip {
        background: #E0F2FE;
        color: #0369A1;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 8px;
        border: 1px solid #BAE6FD;
    }
    .card-summary {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.5;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }
    .card-footer {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid rgba(148,163,184,0.15);
    }
    .view-link {
        font-size: 13px;
        font-weight: 600;
        color: var(--primary) !important;
        text-decoration: none !important;
        transition: color 0.15s ease;
    }
    .view-link:hover { color: var(--primary-dark) !important; }
    .results-label {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
    }

    /* ── Status badge ── */
    .status-badge {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        display: inline-block;
        line-height: 1.5;
        border: 1px solid rgba(0,0,0,0.04);
    }

    /* ── Insight sections ── */
    .insight-section, .query-insight-section {
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid rgba(148,163,184,0.15);
    }
    .insight-label, .query-insight-label {
        font-size: 10.5px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 7px;
    }
    .insight-body, .query-insight-body {
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.75;
    }

    /* ── Intelligence / answer cards ── */
    .centralized-card, .query-answer-card {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-left: 4px solid var(--primary);
        border-radius: var(--radius);
        padding: 22px 26px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        box-shadow: var(--shadow-sm);
    }
    .centralized-icon, .query-answer-icon {
        font-size: 20px;
        line-height: 1;
        flex-shrink: 0;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #E0F2FE;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        box-shadow: none;
    }
    .centralized-content, .query-answer-content { flex: 1; }
    .centralized-header, .query-answer-header {
        font-size: 11px;
        font-weight: 700;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .centralized-body, .query-answer-body {
        font-size: 14.5px;
        color: var(--text-primary);
        line-height: 1.85;
        white-space: pre-wrap;
    }
    .query-sources-label {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 22px 0 14px 0;
    }

    /* ── Rephrase hint ── */
    .rephrase-hint {
        font-size: 12.5px;
        color: var(--text-secondary);
        margin-bottom: 14px;
        padding: 10px 16px;
        background: #F0F9FF;
        border-radius: 10px;
        border: 1px solid #BAE6FD;
    }
    .rephrase-hint em { color: var(--text-primary); font-style: italic; font-weight: 500; }

    /* ── Section headings ── */
    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 20px 0 14px 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Chart containers ── */
    .chart-panel {
        background: var(--surface-muted);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 12px 4px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 8px;
    }

    /* ── AI disclaimer ── */
    .ai-disclaimer {
        font-size: 12px;
        color: var(--text-muted);
        text-align: center;
        margin-top: 12px;
        line-height: 1.55;
        padding: 10px 16px;
        background: #F8FAFC;
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    /* ── Toggle ── */
    .stToggle label span {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
    }

    /* ── Text area ── */
    div[data-testid="stTextArea"] textarea {
        border-radius: var(--radius) !important;
        border: 1.5px solid rgba(148,163,184,0.30) !important;
        font-size: 15px !important;
        background: var(--surface) !important;
        padding: 16px 18px !important;
        line-height: 1.65 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
    }

    /* ── Primary button ── */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(2,132,199,0.25) !important;
        transition: background 0.18s ease, box-shadow 0.18s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--primary-dark) !important;
        box-shadow: 0 4px 12px rgba(2,132,199,0.30) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: none !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border) !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: #BAE6FD !important;
        color: var(--primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ── Metrics — KPI cards ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 18px 20px 14px !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #BAE6FD;
        box-shadow: var(--shadow-md);
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: #0F172A !important;
        background: none;
        -webkit-text-fill-color: #0F172A;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 11.5px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Alerts ── */
    div[data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--primary) !important; }

    /* ── Dividers ── */
    hr {
        border: none;
        height: 1px;
        background: #E2E8F0;
        margin: 1.5rem 0;
    }

    /* ── Caption footer ── */
    .stCaption, small {
        color: var(--text-muted) !important;
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


def _status_badge_colors(status: str):
    """Return (bg_hex, text_hex) for a status pill based on the status name."""
    s = (status or "").lower()
    if any(x in s for x in ("done", "closed", "resolved", "cancelled", "canceled", "rejected")):
        return "#D1FAE5", "#065F46"
    if any(x in s for x in ("investigating", "in progress", "development")):
        return "#FEF3C7", "#92400E"
    if any(x in s for x in ("waiting", "to do", "reopened")):
        return "#FFE4E6", "#9F1239"
    return "#F1F5F9", "#475569"


PROJECT_FULL_NAMES = {
    "SS":    "SaaStars",
    "ACC":   "ATS+CRM+COMM",
    "TITAN": "Titans",
    "TS":    "TechnoStars",
    "RCRM":  "Engineering",
    "AL":    "Alphas",
    "ST":    "Solutions Team",
    "NEO":   "Neo",
    "RDP":   "Revamp Details Pages",
    "BNP":   "Contract Staffing",
    "AT":    "Architecture Team",
}

# ── Jira data loader (cached 30 min) ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_jira_data() -> pd.DataFrame:
    conn = jca.resolve_connection_from_env()
    jql  = (
        "issuetype = 'Bug - Customer Reported' "
        "AND created >= -365d "
        "ORDER BY created DESC"
    )
    rows = []
    for issue in iter_all_issues(conn, jql, max_results=100, fields=list(SEARCH_FIELDS)):
        row = normalize_issue(issue, site_base=conn.api_root)
        # Extract statusCategory directly from raw Jira response — most accurate signal
        f          = issue.get("fields") or {}
        status_obj = f.get("status") or {}
        row["status_category"] = (status_obj.get("statusCategory") or {}).get("name", "")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Dates
    df["created_dt"]   = pd.to_datetime(df["created"],  utc=True, errors="coerce")
    df["resolved_dt"]  = pd.to_datetime(df["resolved"], utc=True, errors="coerce")
    df["month_period"] = df["created_dt"].dt.to_period("M")
    df["month_label"]  = df["created_dt"].dt.strftime("%b %Y")

    # Resolved = statusCategory is "Done" (covers Done, Cancelled, Rejected)
    df["is_resolved"]  = df["status_category"] == "Done"

    # Resolution time in days (only for resolved tickets with both dates)
    df["resolution_days"] = (
        (df["resolved_dt"] - df["created_dt"]).dt.total_seconds() / 86400
    ).where(df["is_resolved"] & df["resolved_dt"].notna())

    # Clean fields
    df["status_clean"] = df["status"].str.strip().fillna("Unknown")
    df["priority"]     = df["priority"].str.strip().replace("", "None").fillna("None")
    df["reporter"]     = df["reporter"].str.strip().replace("", "Unknown").fillna("Unknown")
    df["project_label"]= df["project"].map(PROJECT_FULL_NAMES).fillna(df["project"])

    return df


# ── APP HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1 class="app-title">CSE Ticket Intelligence</h1>
    <p class="app-subtitle">Describe a new defect to find similar existing tickets and probable root causes.</p>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["CSE Tickets", "Analytics"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DUPLICATE FINDER
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    query_mode = st.toggle(
        "Query?",
        value=False,
        help="Turn on to ask a question and get an AI answer from historical CSE queries.",
    )

    if query_mode:
        placeholder = "e.g. Why are some calls charged around $5 when most 10-minute calls cost about $0.40?"
        input_label = "Your Question"
        st.markdown(
            '<p class="mode-hint">Ask a question — we\'ll search past CSE queries and answer using the most relevant tickets.</p>',
            unsafe_allow_html=True,
        )
    else:
        placeholder = "e.g. User sees a blank white screen right after logging in. The CRM dashboard never loads..."
        input_label = "Defect Description"

    query = st.text_area(
        input_label,
        height=120,
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    search_clicked = st.button("Search Knowledge Base", use_container_width=True, type="primary")
    st.markdown(
        '<p class="ai-disclaimer">⚠️ Results are generated using AI. '
        'Use as guidance only — always verify before acting on or sharing with customers.</p>',
        unsafe_allow_html=True,
    )

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a question or defect description before searching.")
        else:
            with st.spinner("Searching knowledge base…"):
                try:
                    if query_mode:
                        output  = find_query_answer(query.strip())
                        results = output.get("sources", [])
                    else:
                        output  = find_duplicates(query.strip())
                        results = output.get("results", [])
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    output  = {}
                    results = []

            if query_mode:
                answer  = output.get("answer", "")
                results = output.get("sources", [])
                if not answer and not results:
                    st.info("No relevant query records found in the knowledge base.")
                else:
                    st.markdown("---")

                    search_q = output.get("search_query", "")
                    if search_q and search_q.strip().lower() != query.strip().lower():
                        st.markdown(
                            f'<p class="rephrase-hint">🔍 Searched as: <em>{_html.escape(search_q)}</em></p>',
                            unsafe_allow_html=True,
                        )

                    if answer:
                        st.markdown(f"""
<div class="query-answer-card">
    <div class="query-answer-icon">💡</div>
    <div class="query-answer-content">
    <div class="query-answer-header">Answer</div>
    <div class="query-answer-body">{_html.escape(answer)}</div>
    </div>
</div>""", unsafe_allow_html=True)
                    elif results:
                        st.info("Found related tickets but could not generate an answer. See sources below.")

                    if results:
                        st.markdown(
                            f'<p class="query-sources-label">Top {len(results)} relevant tickets</p>',
                            unsafe_allow_html=True,
                        )
                        for r in results:
                            key            = r["key"]
                            reporter       = _html.escape(r.get("reporter") or "")
                            summary        = _html.escape(r.get("summary") or "—")
                            status         = r.get("status", "")
                            ticket_summary = _html.escape(r.get("ticket_summary", ""))
                            relevance      = _html.escape(r.get("relevance", ""))
                            combined       = (ticket_summary + (" " if ticket_summary and relevance else "") + relevance).strip()
                            ticket_url     = f"{JIRA_BASE}{key}"
                            bg, fg = _status_badge_colors(status)
                            status_html = (
                                f'<span class="status-badge" style="background:{bg};color:{fg};">'
                                f'● {_html.escape(status)}</span>'
                            ) if status else ""
                            reporter_html = (
                                f'<span class="reporter-chip">👤 {reporter}</span>'
                            ) if reporter else ""
                            insight_html = (
                                f'<div class="query-insight-section">'
                                f'<div class="query-insight-label">AI Summary &amp; Relevance</div>'
                                f'<div class="query-insight-body">{combined}</div>'
                                f'</div>'
                            ) if combined else ""
                            st.markdown(f"""
<div class="query-result-card">
    <div class="card-meta">
    <a class="ticket-chip" href="{ticket_url}" target="_blank">{key}</a>
    {status_html}
    {reporter_html}
    </div>
    <div class="card-summary">{summary}</div>
    {insight_html}
    <div class="card-footer">
    <a class="view-link" href="{ticket_url}" target="_blank">View ticket in Jira →</a>
    </div>
</div>""", unsafe_allow_html=True)

            elif not results:
                st.info("No similar tickets found in the knowledge base.")
            else:
                st.markdown("---")

                # Rephrased query hint
                search_q = output.get("search_query", "")
                if search_q and search_q.strip().lower() != query.strip().lower():
                    st.markdown(
                        f'<p class="rephrase-hint">🔍 Searched as: <em>{_html.escape(search_q)}</em></p>',
                        unsafe_allow_html=True,
                    )

                # Centralized intelligence summary card
                central = output.get("centralized_summary", "")
                if central:
                    st.markdown(f"""
<div class="centralized-card">
    <div class="centralized-icon">💡</div>
    <div class="centralized-content">
    <div class="centralized-header">Intelligence Summary</div>
    <div class="centralized-body">{_html.escape(central)}</div>
    </div>
</div>""", unsafe_allow_html=True)

                st.markdown(f'<p class="results-label">Top {len(results)} matches found</p>', unsafe_allow_html=True)

                for i, r in enumerate(results, 1):
                    key            = r["key"]
                    reporter       = _html.escape(r.get("reporter") or "Unknown")
                    summary        = _html.escape(r.get("summary", "—"))
                    status         = r.get("status", "")
                    ticket_summary = _html.escape(r.get("ticket_summary", ""))
                    insight        = _html.escape(r.get("insight", ""))
                    ticket_url     = f"{JIRA_BASE}{key}"

                    # Status badge
                    bg, fg = _status_badge_colors(status)
                    status_html = (
                        f'<span class="status-badge" style="background:{bg};color:{fg};">'
                        f'● {_html.escape(status)}</span>'
                    ) if status else ""

                    st.markdown(f"""
<div class="result-card">
    <div class="card-meta">
    <a class="ticket-chip" href="{ticket_url}" target="_blank">{key}</a>
    {status_html}
    <span class="reporter-chip">👤 {reporter}</span>
    </div>
    <div class="card-summary">{summary}</div>
    <div class="insight-section">
    <div class="insight-label">AI Ticket Summary</div>
    <div class="insight-body">{ticket_summary}</div>
    </div>
    <div class="insight-section">
    <div class="insight-label">Similarity Insight &amp; RCA</div>
    <div class="insight-body">{insight}</div>
    </div>
    <div class="card-footer">
    <a class="view-link" href="{ticket_url}" target="_blank">View ticket in Jira →</a>
    </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="page-header">
        <p class="page-title">Ticket Analytics</p>
        <p class="page-subtitle">Customer-reported bugs — last 12 months · auto-refreshes every 30 min</p>
    </div>
    """, unsafe_allow_html=True)

    col_ref, _ = st.columns([1, 6])
    with col_ref:
        if st.button("Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading ticket data from Jira…"):
        try:
            df = load_jira_data()
            load_error = None
        except Exception as exc:
            df = pd.DataFrame()
            load_error = str(exc)

    if load_error:
        st.error(f"Failed to load Jira data: {load_error}")
        st.stop()

    if df.empty:
        st.info("No tickets found for the selected period.")
        st.stop()

    # ── Derived counts (using statusCategory — authoritative) ─────────────────
    total       = len(df)
    resolved    = int(df["is_resolved"].sum())                          # statusCategory == "Done"
    in_progress = int((df["status_category"] == "In Progress").sum())  # Investigating etc.
    pending     = int((df["status_category"] == "To Do").sum())        # Waiting + To Do
    pct_closed  = round(resolved / total * 100, 1) if total else 0
    avg_res_days= df["resolution_days"].dropna()
    avg_res     = f"{avg_res_days.mean():.1f}d" if not avg_res_days.empty else "—"

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Tickets",      f"{total:,}",       "Last 12 months")
    k2.metric("Pending / Waiting",  f"{pending:,}",     "To Do category")
    k3.metric("In Progress",        f"{in_progress:,}", "Being investigated")
    k4.metric("Done / Cancelled",   f"{resolved:,}",    f"{pct_closed:.1f}% resolution rate")
    k5.metric("Avg Resolution Time",avg_res,            f"{df['project'].nunique()} projects")

    st.markdown("---")

    # ── Row 1: Monthly trend + Status breakdown ───────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-title">📅 Monthly Ticket Trend</p>', unsafe_allow_html=True)
        monthly = (
            df.groupby("month_period")
              .size()
              .reset_index(name="count")
              .sort_values("month_period")
        )
        monthly["label"] = monthly["month_period"].dt.strftime("%b %Y")

        fig_trend = px.bar(
            monthly, x="label", y="count",
            color_discrete_sequence=[CHART_PRIMARY],
            labels={"label": "", "count": "Tickets"},
        )
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(tickangle=-35, showgrid=False, linecolor="rgba(148,163,184,0.3)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            showlegend=False,
            bargap=0.28,
        )
        fig_trend.update_traces(
            marker_line_width=0,
            marker_color=CHART_PRIMARY,
            hovertemplate="<b>%{x}</b><br>%{y} tickets<extra></extra>",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-title">🟢 Status Breakdown</p>', unsafe_allow_html=True)
        status_counts = df["status_clean"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]

        colors = [STATUS_COLORS.get(s, "#A0AEC0") for s in status_counts["status"]]

        fig_status = go.Figure(go.Pie(
            labels=status_counts["status"],
            values=status_counts["count"],
            hole=0.52,
            marker=dict(colors=colors, line=dict(color="#fff", width=2)),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} tickets (%{percent})<extra></extra>",
        ))
        fig_status.update_layout(
            **PLOTLY_LAYOUT,
            legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=11)),
            showlegend=True,
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # ── Row 2: Priority + Top Reporters ──────────────────────────────────────
    col_pri, col_rep = st.columns(2)

    with col_pri:
        st.markdown('<p class="section-title">⚡ Priority Distribution</p>', unsafe_allow_html=True)
        pri_counts = df["priority"].value_counts().reset_index()
        pri_counts.columns = ["priority", "count"]

        fig_pri = px.bar(
            pri_counts, x="count", y="priority",
            orientation="h",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            labels={"count": "Tickets", "priority": ""},
            text="count",
        )
        fig_pri.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        fig_pri.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_pri, use_container_width=True)

    with col_rep:
        st.markdown('<p class="section-title">👤 Top 10 Reporters</p>', unsafe_allow_html=True)
        rep_counts = (
            df["reporter"].value_counts()
              .head(10)
              .reset_index()
        )
        rep_counts.columns = ["reporter", "count"]
        rep_counts = rep_counts.sort_values("count")

        fig_rep = px.bar(
            rep_counts, x="count", y="reporter",
            orientation="h",
            color_discrete_sequence=[CHART_SECONDARY],
            labels={"count": "Tickets", "reporter": ""},
            text="count",
        )
        fig_rep.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        fig_rep.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_rep, use_container_width=True)

    # ── Row 3: Project breakdown ──────────────────────────────────────────────
    st.markdown('<p class="section-title">🗂️ Tickets by Project</p>', unsafe_allow_html=True)

    proj_counts = df.groupby(["project", "project_label"]).size().reset_index(name="count")
    proj_counts = proj_counts.sort_values("count", ascending=False)

    proj_resolved = (
        df[df["is_resolved"]].groupby("project").size().reset_index(name="resolved")
    )
    proj_counts = proj_counts.merge(proj_resolved, on="project", how="left").fillna(0)
    proj_counts["resolved"] = proj_counts["resolved"].astype(int)
    proj_counts["open"]     = proj_counts["count"] - proj_counts["resolved"]
    proj_counts["resolution_rate"] = (proj_counts["resolved"] / proj_counts["count"] * 100).round(1)
    proj_counts["label"] = proj_counts.apply(
        lambda r: f"{r['project_label']} ({r['project']})", axis=1
    )

    col_proj_bar, col_proj_rate = st.columns([3, 2])

    with col_proj_bar:
        fig_proj = px.bar(
            proj_counts, x="label", y="count",
            color="label",
            color_discrete_sequence=PROJECT_COLORS,
            labels={"label": "", "count": "Tickets"},
            text="count",
        )
        fig_proj.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=False, tickangle=-25),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            showlegend=False,
        )
        fig_proj.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_proj, use_container_width=True)

    with col_proj_rate:
        st.markdown('<p class="section-title">✅ Resolution Rate by Project</p>', unsafe_allow_html=True)
        proj_sorted = proj_counts.sort_values("resolution_rate")
        fig_rate = px.bar(
            proj_sorted,
            x="resolution_rate", y="label",
            orientation="h",
            color="resolution_rate",
            color_continuous_scale=["#F43F5E", "#F59E0B", "#10B981"],
            range_color=[0, 100],
            labels={"resolution_rate": "Resolved %", "label": ""},
            text=proj_sorted["resolution_rate"].apply(lambda x: f"{x}%"),
        )
        fig_rate.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", range=[0, 110], zeroline=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
            coloraxis_showscale=False,
        )
        fig_rate.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig_rate, use_container_width=True)

    # ── Row 4: Avg resolution time by priority + Status category breakdown ────
    col_restime, col_cat = st.columns(2)

    with col_restime:
        st.markdown('<p class="section-title">⏱️ Avg Resolution Time by Priority</p>', unsafe_allow_html=True)
        PRIORITY_ORDER = ["Blocker", "Critical", "High", "Medium", "Low", "Lowest"]
        res_time = (
            df.dropna(subset=["resolution_days"])
              .groupby("priority")["resolution_days"]
              .agg(avg="mean", median="median", count="count")
              .reset_index()
        )
        res_time["avg"]    = res_time["avg"].round(1)
        res_time["median"] = res_time["median"].round(1)
        res_time["priority_order"] = pd.Categorical(
            res_time["priority"], categories=PRIORITY_ORDER, ordered=True
        )
        res_time = res_time.sort_values("priority_order")

        fig_res = px.bar(
            res_time, x="priority", y="avg",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            labels={"priority": "", "avg": "Avg days to resolve"},
            text=res_time["avg"].apply(lambda x: f"{x}d"),
            custom_data=["median", "count"],
        )
        fig_res.update_traces(
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Avg: %{y}d<br>Median: %{customdata[0]}d<br>Tickets: %{customdata[1]}<extra></extra>",
        )
        fig_res.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.15)", title="Days", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_cat:
        st.markdown('<p class="section-title">📂 Tickets by Status Category</p>', unsafe_allow_html=True)
        cat_counts = df["status_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        CAT_COLORS = {"Done": "#10B981", "In Progress": "#F59E0B", "To Do": "#F43F5E"}
        cat_colors = [CAT_COLORS.get(c, "#94A3B8") for c in cat_counts["category"]]

        fig_cat = go.Figure(go.Pie(
            labels=cat_counts["category"],
            values=cat_counts["count"],
            hole=0.55,
            marker=dict(colors=cat_colors, line=dict(color="#fff", width=3)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} tickets (%{percent})<extra></extra>",
        ))
        fig_cat.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if "created_dt" in df.columns and df["created_dt"].notna().any():
        newest = df["created_dt"].max().strftime("%d %b %Y %H:%M UTC")
        oldest = df["created_dt"].min().strftime("%d %b %Y")
        st.caption(f"Data range: {oldest} → {newest}  ·  {total:,} tickets loaded")
