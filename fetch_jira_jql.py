#!/usr/bin/env python3
"""
Fetch Jira issues via POST /rest/api/3/search/jql and return structured fields.

Mirrors the enhanced JQL search API (fields, pagination). Auth comes from .env via
jira_cloud_auth (JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN).

Usage:
  python3 fetch_jira_jql.py
  python3 fetch_jira_jql.py --output issues.json
  python3 fetch_jira_jql.py --format csv --output issues.csv
  JIRA_JQL='project = CSE AND ...' python3 fetch_jira_jql.py

API:
  https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-search/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import jira_cloud_auth as jca

DEFAULT_JQL = (
    "issuetype = 'Bug - Customer Reported' "
    "AND created >= '2025-06-01' "
    "AND created <= '2026-06-14' "
    "ORDER BY updated DESC"
)

SEARCH_FIELDS = [
    "summary",
    "description",
    "status",
    "reporter",
    "assignee",
    "priority",
    "issuetype",
    "created",
    "updated",
    "resolutiondate",
    "labels",
    "comment",
    "project",
]

OUTPUT_COLUMNS = [
    "issue_key",
    "summary",
    "description",
    "status",
    "reporter",
    "assignee",
    "priority",
    "issue_type",
    "project",
    "created",
    "updated",
    "resolved",
    "labels",
    "comments",
    "url",
]

MAX_RESULTS = 100


def adf_to_text(node: Any) -> str:
    """Convert Atlassian Document Format (ADF) to plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type")
    if node_type == "text":
        return str(node.get("text") or "")

    content = node.get("content") or []
    child_text = "".join(adf_to_text(child) for child in content)

    if node_type in ("paragraph", "heading", "blockquote", "panel"):
        return child_text + "\n"
    if node_type == "hardBreak":
        return "\n"
    if node_type in ("bulletList", "orderedList"):
        return child_text
    if node_type == "listItem":
        return "- " + child_text.strip() + "\n"
    if node_type == "codeBlock":
        return child_text + "\n"
    if node_type == "rule":
        return "\n---\n"
    if node_type == "mention":
        return str(node.get("attrs", {}).get("text") or child_text or "@mention")
    if node_type == "emoji":
        return str(node.get("attrs", {}).get("shortName") or "")
    if node_type == "media":
        alt = (node.get("attrs") or {}).get("alt")
        return f"[media: {alt}]" if alt else "[media]"
    if node_type == "table":
        return child_text + "\n"
    if node_type in ("tableRow", "tableHeader", "tableCell"):
        return child_text + " "
    if node_type == "doc":
        return child_text.strip()

    return child_text


def _user_name(user: Any) -> str:
    if not isinstance(user, dict):
        return ""
    return str(
        user.get("displayName")
        or user.get("emailAddress")
        or user.get("accountId")
        or ""
    ).strip()


def _named_field(field: Any, key: str = "name") -> str:
    if isinstance(field, dict):
        return str(field.get(key) or "").strip()
    return ""


def _iso_to_display(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d/%b/%y %I:%M %p")
    except ValueError:
        return text


def format_comments(comment_field: Any) -> str:
    if not isinstance(comment_field, dict):
        return ""
    comments = comment_field.get("comments") or []
    if not isinstance(comments, list):
        return ""

    blocks: list[str] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        created = _iso_to_display(comment.get("created"))
        author = _user_name(comment.get("author"))
        body = adf_to_text(comment.get("body"))
        if not body.strip():
            continue
        header = f"{created};{author};{body.strip()}"
        blocks.append(header)

    return "\n\n---\n\n".join(blocks)


def normalize_issue(issue: dict, *, site_base: str) -> dict[str, str]:
    key = str(issue.get("key") or "").strip()
    fields = issue.get("fields") or {}
    site = site_base.rstrip("/")

    description = fields.get("description")
    if isinstance(description, dict):
        description_text = adf_to_text(description)
    else:
        description_text = str(description or "").strip()

    labels = fields.get("labels") or []
    if isinstance(labels, list):
        labels_text = ", ".join(str(label) for label in labels if str(label).strip())
    else:
        labels_text = ""

    return {
        "issue_key": key,
        "summary": str(fields.get("summary") or "").strip(),
        "description": description_text,
        "status": _named_field(fields.get("status")),
        "reporter": _user_name(fields.get("reporter")),
        "assignee": _user_name(fields.get("assignee")),
        "priority": _named_field(fields.get("priority")),
        "issue_type": _named_field(fields.get("issuetype")),
        "project": _named_field(fields.get("project"), key="key"),
        "created": str(fields.get("created") or "").strip(),
        "updated": str(fields.get("updated") or "").strip(),
        "resolved": str(fields.get("resolutiondate") or "").strip(),
        "labels": labels_text,
        "comments": format_comments(fields.get("comment")),
        "url": f"{site}/browse/{key}" if key else "",
    }


def search_jql_page(
    conn: jca.JiraConnection,
    jql: str,
    *,
    max_results: int,
    next_page_token: str | None,
    fields: list[str],
) -> dict:
    payload: dict[str, Any] = {
        "jql": jql,
        "maxResults": max_results,
        "fields": fields,
        "fieldsByKeys": True,
        "expand": "names",
    }
    if next_page_token:
        payload["nextPageToken"] = next_page_token

    body = json.dumps(payload).encode("utf-8")
    url = conn.api_root.rstrip("/") + jca.SEARCH_REL
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": conn.auth_header,
    }
    try:
        _, data = jca.http_json("POST", url, headers, body=body, timeout=120)
    except jca.HttpJiraError as e:
        jca.print_oauth_scope_hint_on_forbidden(e.code, e.body)
        jca.print_jira_api_error(e.code, e.reason, e.body)
        raise SystemExit(1) from e
    if not isinstance(data, dict):
        raise SystemExit("Unexpected search response (not a JSON object).")
    return data


def iter_all_issues(
    conn: jca.JiraConnection,
    jql: str,
    *,
    max_results: int,
    fields: list[str],
):
    next_token: str | None = None
    while True:
        page = search_jql_page(
            conn,
            jql,
            max_results=max_results,
            next_page_token=next_token,
            fields=fields,
        )
        issues = page.get("issues") or []
        for issue in issues:
            if isinstance(issue, dict):
                yield issue
        next_token = (page.get("nextPageToken") or "").strip() or None
        if not next_token:
            break


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Jira issues via JQL (POST search).")
    parser.add_argument(
        "--jql",
        help="JQL query (overrides JIRA_JQL env and script default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write results to this file (.json or .csv). Prints JSON to stdout if omitted.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format when --output is set (default: json)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=MAX_RESULTS,
        help=f"Page size for search (default: {MAX_RESULTS})",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help='Request fields=["*all"] instead of the curated field list',
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress auth / progress messages on stderr",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=True)

    conn = jca.resolve_connection_from_env()
    jql = (
        (args.jql or os.environ.get("JIRA_JQL") or "").strip().strip('"').strip("'")
        or DEFAULT_JQL
    )
    fields = ["*all"] if args.all_fields else list(SEARCH_FIELDS)
    site_base = (
        (os.environ.get("JIRA_BASE_URL") or conn.api_root).strip().strip('"').strip("'")
    )

    if not args.quiet:
        print("JQL:", jql, file=sys.stderr)
        print("Search: POST /rest/api/3/search/jql", file=sys.stderr)

    jca.verify_connection(conn, quiet=args.quiet)

    rows: list[dict[str, str]] = []
    for issue in iter_all_issues(
        conn,
        jql,
        max_results=max(1, args.max_results),
        fields=fields,
    ):
        rows.append(normalize_issue(issue, site_base=site_base))

    if not args.quiet:
        print(f"Fetched {len(rows)} issue(s).", file=sys.stderr)

    if args.output:
        out = args.output
        fmt = args.format
        if out.suffix.lower() == ".csv":
            fmt = "csv"
        elif out.suffix.lower() == ".json":
            fmt = "json"
        if fmt == "csv":
            write_csv(out, rows)
        else:
            write_json(out, rows)
        if not args.quiet:
            print(f"Wrote {len(rows)} issue(s) to {out}", file=sys.stderr)
        return

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
