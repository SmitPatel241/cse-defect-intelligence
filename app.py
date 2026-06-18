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

from query import find_duplicates
import jira_cloud_auth as jca
from fetch_jira_jql import iter_all_issues, normalize_issue, SEARCH_FIELDS

JIRA_BASE   = "https://rcrm.atlassian.net/servicedesk/customer/portal/2/"
JIRA_BROWSE = "https://rcrm.atlassian.net/browse/"

RESOLVED_STATUSES = {
    "done", "closed", "resolved", "complete", "fixed",
    "cancelled", "canceled", "rejected", "duplicate",
    "won't fix", "wont fix", "cannot reproduce",
}

STATUS_COLORS = {
    "Done":                      "#68D391",
    "Cancelled":                 "#A0AEC0",
    "Rejected":                  "#CBD5E0",
    "To Do":                     "#FC8181",
    "Investigating":             "#F6AD55",
    "Development In Progress":   "#F6AD55",
    "Reopened":                  "#FC8181",
    "Waiting for customer":      "#63B3ED",
    "Waiting For CSA":           "#63B3ED",
    "Waiting For Third Party":   "#B794F4",
    "Waiting For Engineer":      "#B794F4",
    "Waiting for Product Team":  "#B794F4",
}

PRIORITY_COLORS = {
    "Blocker":  "#742A2A",
    "Critical": "#FC8181",
    "High":     "#F6AD55",
    "Medium":   "#63B3ED",
    "Low":      "#68D391",
    "Lowest":   "#C6F6D5",
    "None":     "#CBD5E0",
}

PROJECT_COLORS = [
    "#4F6EF7","#7C8FF5","#63B3ED","#48BB78","#F6AD55",
    "#FC8181","#B794F4","#68D391","#F687B3","#76E4F7","#FBD38D",
]

st.set_page_config(
    page_title="CSE Ticket Intelligence",
    page_icon="🎯",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F5F7FA; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* ── Duplicate-finder card ── */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .card-meta { display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap; }
    .ticket-chip {
        background:#4F6EF7; color:#fff !important;
        font-size:12.5px; font-weight:700;
        padding:4px 12px; border-radius:6px;
        text-decoration:none !important; letter-spacing:.02em;
    }
    .ticket-chip:hover { background:#3B55D9; }
    .reporter-chip {
        background:#EEF2FF; color:#4338CA;
        font-size:12px; font-weight:500;
        padding:4px 10px; border-radius:6px;
    }
    .card-summary { font-size:16px; font-weight:600; color:#1A202C; line-height:1.45; margin-bottom:14px; }
    .card-footer { margin-top:14px; padding-top:12px; border-top:1px solid #F0F4F8; }
    .view-link  { font-size:13px; font-weight:600; color:#4F6EF7 !important; text-decoration:none !important; }
    .view-link:hover { text-decoration:underline !important; }
    .results-label { font-size:13px; font-weight:600; color:#718096; text-transform:uppercase; letter-spacing:.06em; margin-bottom:16px; }

    /* ── Status badge ── */
    .status-badge {
        font-size:11.5px; font-weight:600;
        padding:3px 10px; border-radius:20px;
        display:inline-block; line-height:1.6;
    }

    /* ── Per-ticket insight sections ── */
    .insight-section { margin-top:14px; padding-top:12px; border-top:1px solid #F0F4F8; }
    .insight-label   { font-size:11px; font-weight:700; color:#A0AEC0; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
    .insight-body    { font-size:14px; color:#4A5568; line-height:1.75; }

    /* ── Centralized intelligence summary card ── */
    .centralized-card {
        background: linear-gradient(135deg,#EBF4FF 0%,#F0F4FF 100%);
        border:1px solid #C3DAFE; border-left:4px solid #4F6EF7;
        border-radius:12px; padding:18px 22px; margin-bottom:18px;
        display:flex; align-items:flex-start; gap:14px;
    }
    .centralized-icon    { font-size:20px; line-height:1.6; flex-shrink:0; }
    .centralized-content { flex:1; }
    .centralized-header  { font-size:11px; font-weight:700; color:#4F6EF7; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
    .centralized-body    { font-size:14px; color:#2D3748; line-height:1.75; }

    /* ── Rephrased query hint ── */
    .rephrase-hint { font-size:12px; color:#718096; margin-bottom:12px; }
    .rephrase-hint em { color:#4A5568; font-style:italic; }

    /* ── KPI boxes ── */
    .kpi-row { display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }
    .kpi-box {
        flex:1; min-width:160px;
        background:#fff; border:1px solid #E2E8F0;
        border-radius:12px; padding:20px 22px;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-label { font-size:12px; font-weight:600; color:#718096; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
    .kpi-value { font-size:32px; font-weight:700; color:#1A202C; line-height:1; }
    .kpi-sub   { font-size:12px; color:#A0AEC0; margin-top:4px; }

    /* ── Section headings ── */
    .section-title { font-size:16px; font-weight:700; color:#2D3748; margin:24px 0 12px 0; }

    /* Input */
    div[data-testid="stTextArea"] textarea {
        border-radius:10px !important; border:1.5px solid #CBD5E0 !important;
        font-size:15px !important; background:#fff !important;
        padding:12px !important; line-height:1.6 !important;
    }
    div[data-testid="stButton"] > button {
        border-radius:10px !important; font-size:15px !important;
        font-weight:600 !important; padding:.65rem 1.5rem !important;
        background:#4F6EF7 !important; color:white !important;
        border:none !important; transition:opacity .15s ease;
    }
    div[data-testid="stButton"] > button:hover { opacity:.88 !important; }
    hr { border-color:#E2E8F0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


def _status_badge_colors(status: str):
    """Return (bg_hex, text_hex) for a status pill based on the status name."""
    s = (status or "").lower()
    if any(x in s for x in ("done", "closed", "resolved", "cancelled", "canceled", "rejected")):
        return "#C6F6D5", "#276749"   # green
    if any(x in s for x in ("investigating", "in progress", "development")):
        return "#FEEBC8", "#7B341E"   # orange
    if any(x in s for x in ("waiting", "to do", "reopened")):
        return "#FED7D7", "#742A2A"   # red/pink
    return "#EDF2F7", "#4A5568"       # neutral gray


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


# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 CSE Tickets", "📊 Analytics"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DUPLICATE FINDER
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    _, col, _ = st.columns([1, 5, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.8rem;">
            <p style="font-size:26px;font-weight:700;color:#1A202C;margin:0 0 6px 0;">🎯 CSE Ticket Intelligence</p>
            <p style="font-size:15px;color:#718096;margin:0;">Describe a new defect to find similar existing tickets and probable root causes.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        query = st.text_area(
            "Defect Description",
            height=120,
            placeholder="e.g. User sees a blank white screen right after logging in. The CRM dashboard never loads...",
            label_visibility="collapsed",
        )
        search_clicked = st.button("🔍  Search Knowledge Base", use_container_width=True)

        if search_clicked:
            if not query.strip():
                st.warning("Please enter a defect description before searching.")
            else:
                with st.spinner("Searching knowledge base…"):
                    try:
                        output  = find_duplicates(query.strip())
                        results = output.get("results", [])
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")
                        output  = {}
                        results = []

                if not results:
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
    <p style="font-size:22px;font-weight:700;color:#1A202C;margin:0 0 4px 0;">📊 Ticket Analytics</p>
    <p style="font-size:14px;color:#718096;margin:0 0 20px 0;">Customer-reported bugs — last 12 months · auto-refreshes every 30 min</p>
    """, unsafe_allow_html=True)

    col_ref, _ = st.columns([1, 6])
    with col_ref:
        if st.button("🔄 Refresh Data"):
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
            color_discrete_sequence=["#4F6EF7"],
            labels={"label": "", "count": "Tickets"},
        )
        fig_trend.update_layout(
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            font=dict(family="sans-serif", size=12),
            showlegend=False,
            bargap=0.25,
        )
        fig_trend.update_traces(marker_line_width=0)
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
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="sans-serif", size=12),
            legend=dict(orientation="v", x=1.0, y=0.5),
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
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(showgrid=False),
            font=dict(family="sans-serif", size=12),
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
            color_discrete_sequence=["#7C8FF5"],
            labels={"count": "Tickets", "reporter": ""},
            text="count",
        )
        fig_rep.update_layout(
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(showgrid=False),
            font=dict(family="sans-serif", size=12),
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
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            font=dict(family="sans-serif", size=12),
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
            color_continuous_scale=["#FC8181", "#F6AD55", "#68D391"],
            range_color=[0, 100],
            labels={"resolution_rate": "Resolved %", "label": ""},
            text=proj_sorted["resolution_rate"].apply(lambda x: f"{x}%"),
        )
        fig_rate.update_layout(
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8", range=[0, 110]),
            yaxis=dict(showgrid=False),
            font=dict(family="sans-serif", size=12),
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
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="Days"),
            font=dict(family="sans-serif", size=12),
            showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_cat:
        st.markdown('<p class="section-title">📂 Tickets by Status Category</p>', unsafe_allow_html=True)
        cat_counts = df["status_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        CAT_COLORS = {"Done": "#68D391", "In Progress": "#F6AD55", "To Do": "#FC8181"}
        cat_colors = [CAT_COLORS.get(c, "#A0AEC0") for c in cat_counts["category"]]

        fig_cat = go.Figure(go.Pie(
            labels=cat_counts["category"],
            values=cat_counts["count"],
            hole=0.55,
            marker=dict(colors=cat_colors, line=dict(color="#fff", width=3)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} tickets (%{percent})<extra></extra>",
        ))
        fig_cat.update_layout(
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="sans-serif", size=13),
            showlegend=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if "created_dt" in df.columns and df["created_dt"].notna().any():
        newest = df["created_dt"].max().strftime("%d %b %Y %H:%M UTC")
        oldest = df["created_dt"].min().strftime("%d %b %Y")
        st.caption(f"Data range: {oldest} → {newest}  ·  {total:,} tickets loaded")
