"""
Shared Jira Cloud authentication for scripts.

OAuth 2.0 (3LO) apps request scopes in the developer console; classic **read:jira-work**
covers reading projects/issues and **search for issues** (REST issue search / JQL). Scopes do
not replace Jira project permissions (e.g. Browse projects).

Scopes reference:
https://developer.atlassian.com/cloud/jira/platform/scopes-for-oauth-2-3LO-and-forge-apps/

OAuth API base (Bearer):
https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/#other-integrations

Personal API tokens (Basic to site URL):
https://developer.atlassian.com/cloud/jira/platform/basic-auth-for-rest-apis/

**API tokens with scopes** (Create API token with scopes): still use **HTTP Basic** (email + token),
but requests must go to **https://api.atlassian.com/ex/jira/{cloudId}/...**, not your site URL.
See Atlassian Account docs (same URL pattern as 3LO step 3.2; auth is Basic, not Bearer):
https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode, urlparse

import certifi

SCOPES_FOR_OAUTH_DOC = (
    "https://developer.atlassian.com/cloud/jira/platform/scopes-for-oauth-2-3LO-and-forge-apps/"
)
OAUTH_3LO_APPS_DOC = "https://developer.atlassian.com/cloud/jira/platform/oauth-2-3lo-apps/"

SEARCH_REL = "/rest/api/3/search/jql"
ACCESSIBLE_RESOURCES_URL = "https://api.atlassian.com/oauth/token/accessible-resources"
ATLASSIAN_ME_URL = "https://api.atlassian.com/me"

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class HttpJiraError(Exception):
    def __init__(self, code: int, reason: str, body: str) -> None:
        super().__init__(f"HTTP {code} {reason}")
        self.code = code
        self.reason = reason
        self.body = body


def _jira_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
    )


def http_json(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
    timeout: int = 120,
) -> tuple[int, dict | list]:
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers,
    )
    opener = _jira_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise HttpJiraError(e.code, e.reason, err_body) from None


def print_jira_api_error(status: int, reason: str, body: str) -> None:
    print(f"Jira API error: HTTP {status} {reason}", file=sys.stderr)
    if not body.strip():
        print("(empty response body)", file=sys.stderr)
        return
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print("Response body (not JSON):", file=sys.stderr)
        print(body, file=sys.stderr)
        return
    if not isinstance(data, dict):
        print(json.dumps(data, indent=2), file=sys.stderr)
        return
    messages = data.get("errorMessages") or []
    field_errors = data.get("errors") or {}
    if messages:
        print("errorMessages:", file=sys.stderr)
        for m in messages:
            print(f"  - {m}", file=sys.stderr)
    if isinstance(field_errors, dict) and field_errors:
        print("errors:", file=sys.stderr)
        for key, val in field_errors.items():
            print(f"  - {key}: {val}", file=sys.stderr)
    if not messages and not field_errors:
        print("Full JSON body:", file=sys.stderr)
        print(json.dumps(data, indent=2), file=sys.stderr)


def print_oauth_scope_hint_on_forbidden(status: int, body: str) -> None:
    if status != 403:
        return
    low = body.lower()
    if "scope" not in low and "forbidden" not in low and "permission" not in low:
        return
    print(
        "If this is an OAuth (3LO) token: ensure the app includes the classic scope **read:jira-work** "
        "for issue/JQL search (and **read:jira-user** if you call GET /rest/api/3/myself). "
        "Granular scopes must match each REST operation. Scopes do not bypass Jira Browse projects.",
        file=sys.stderr,
    )
    print(f"See: {SCOPES_FOR_OAUTH_DOC}", file=sys.stderr)


def auth_basic_header(email: str, api_token: str) -> str:
    pair = f"{email}:{api_token}".encode("utf-8")
    return "Basic " + base64.b64encode(pair).decode("ascii")


def normalize_site_url(url: str) -> str:
    return url.rstrip("/").lower()


def looks_like_jira_personal_api_token(token: str) -> bool:
    t = token.strip()
    return len(t) >= 8 and t.upper().startswith("ATATT")


def print_oauth_bearer_rejected_help(access_token: str, context: str) -> None:
    print(context, file=sys.stderr)
    if looks_like_jira_personal_api_token(access_token):
        print(
            "This value looks like a Jira *personal* API token. Use Basic auth to "
            "https://<site>.atlassian.net (JIRA_AUTH_TYPE unset or basic), not Bearer to api.atlassian.com.",
            file=sys.stderr,
        )
        return
    print(
        "JIRA_ACCESS_TOKEN must be an OAuth 2.0 *access* token from the 3LO authorization-code flow.",
        file=sys.stderr,
    )
    print(f"See: {OAUTH_3LO_APPS_DOC}", file=sys.stderr)


def resolve_cloud_id_from_site_basic(email: str, api_token: str, site_base_url: str) -> str:
    """Resolve cloud id using Basic auth (scoped API tokens). Same JSON shape as Bearer flow."""
    headers = {
        "Accept": "application/json",
        "Authorization": auth_basic_header(email, api_token),
    }
    try:
        _, data = http_json("GET", ACCESSIBLE_RESOURCES_URL, headers, timeout=60)
    except HttpJiraError as e:
        print(
            "Could not list accessible resources with Basic auth (scoped token flow).",
            file=sys.stderr,
        )
        print(
            "Set **JIRA_CLOUD_ID** in .env to your Jira cloud UUID, then retry. "
            "Find it in Atlassian Admin (your site) or from your org’s site/product settings; "
            "it matches the {cloudId} in https://api.atlassian.com/ex/jira/{cloudId}/…",
            file=sys.stderr,
        )
        print(f"Account / token help: https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/", file=sys.stderr)
        print_jira_api_error(e.code, e.reason, e.body)
        raise SystemExit(1) from e
    if not isinstance(data, list):
        print("Unexpected accessible-resources response (not a list).", file=sys.stderr)
        print(json.dumps(data, indent=2)[:2000], file=sys.stderr)
        raise SystemExit(1)
    want = normalize_site_url(site_base_url)
    want_host = (urlparse(site_base_url).hostname or "").lower()
    for r in data:
        ru = (r.get("url") or "").strip()
        if not ru:
            continue
        if normalize_site_url(ru) == want:
            return str(r["id"])
    for r in data:
        ru = (r.get("url") or "").strip()
        host = (urlparse(ru).hostname or "").lower()
        if want_host and host == want_host:
            return str(r["id"])
    print(
        "No site matched JIRA_BASE_URL in accessible-resources. Set JIRA_CLOUD_ID explicitly.",
        file=sys.stderr,
    )
    for r in data:
        print(f"  - {r.get('name', '')!r}  id={r.get('id')}  url={r.get('url')}", file=sys.stderr)
    raise SystemExit(1)


def resolve_cloud_id_from_site(access_token: str, site_base_url: str) -> str:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    try:
        _, data = http_json("GET", ACCESSIBLE_RESOURCES_URL, headers, timeout=60)
    except HttpJiraError as e:
        print("Could not list accessible OAuth resources.", file=sys.stderr)
        if e.code == 401:
            print_oauth_bearer_rejected_help(
                access_token,
                "HTTP 401: Atlassian did not accept this string as an OAuth Bearer access token.",
            )
        print_jira_api_error(e.code, e.reason, e.body)
        raise SystemExit(1) from e
    if not isinstance(data, list):
        print("Unexpected accessible-resources response (not a list).", file=sys.stderr)
        print(json.dumps(data, indent=2)[:2000], file=sys.stderr)
        raise SystemExit(1)
    want = normalize_site_url(site_base_url)
    want_host = (urlparse(site_base_url).hostname or "").lower()
    for r in data:
        ru = (r.get("url") or "").strip()
        if not ru:
            continue
        if normalize_site_url(ru) == want:
            return str(r["id"])
    for r in data:
        ru = (r.get("url") or "").strip()
        host = (urlparse(ru).hostname or "").lower()
        if want_host and host == want_host:
            return str(r["id"])
    print(
        "No OAuth cloud matched JIRA_BASE_URL. Set JIRA_CLOUD_ID or fix JIRA_BASE_URL to a listed url:",
        file=sys.stderr,
    )
    for r in data:
        print(f"  - {r.get('name', '')!r}  id={r.get('id')}  url={r.get('url')}", file=sys.stderr)
    raise SystemExit(1)


@dataclass(frozen=True)
class JiraConnection:
    api_root: str
    auth_header: str
    mode: Literal["basic", "oauth", "scoped"]


def resolve_basic_connection_from_env() -> JiraConnection:
    """
    Personal / classic API token: HTTP Basic (email + token) against JIRA_BASE_URL.

    https://developer.atlassian.com/cloud/jira/platform/basic-auth-for-rest-apis/
    """
    base = (os.environ.get("JIRA_BASE_URL") or "").strip().strip('"').strip("'")
    email = (os.environ.get("JIRA_EMAIL") or "").strip().strip('"').strip("'")
    api_token = (
        (os.environ.get("JIRA_API_TOKEN") or os.environ.get("JIRA_ACCESS_TOKEN") or "")
        .strip()
        .strip('"')
        .strip("'")
    )
    if not base or not email or not api_token:
        print(
            "Basic auth requires JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN in .env",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return JiraConnection(
        api_root=base.rstrip("/"),
        auth_header=auth_basic_header(email, api_token),
        mode="basic",
    )


def resolve_connection_from_env() -> JiraConnection:
    """Build api root + Authorization header from JIRA_* env (see module docstring)."""
    raw_auth = (os.environ.get("JIRA_AUTH_TYPE") or "basic").strip().lower()
    scoped_flag = (os.environ.get("JIRA_USE_SCOPED_API_TOKEN") or "").strip() in ("1", "true", "yes")
    token_candidate = (
        (os.environ.get("JIRA_ACCESS_TOKEN") or os.environ.get("JIRA_API_TOKEN") or "")
        .strip()
        .strip('"')
        .strip("'")
    )
    email_present = bool((os.environ.get("JIRA_EMAIL") or "").strip().strip('"').strip("'"))

    # Scoped API token: Basic auth, but base URL must be api.atlassian.com/ex/jira/{cloudId}/…
    if raw_auth in ("scoped", "scoped-basic", "api-token-scoped") or scoped_flag:
        site_url = (os.environ.get("JIRA_BASE_URL") or "").strip().strip('"').strip("'")
        email = (os.environ.get("JIRA_EMAIL") or "").strip().strip('"').strip("'")
        api_token = (
            (os.environ.get("JIRA_API_TOKEN") or os.environ.get("JIRA_ACCESS_TOKEN") or "")
            .strip()
            .strip('"')
            .strip("'")
        )
        if not site_url or not email or not api_token:
            print(
                "Scoped API token mode requires JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN "
                "(or JIRA_ACCESS_TOKEN).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        cloud_id = (os.environ.get("JIRA_CLOUD_ID") or "").strip().strip('"').strip("'")
        if not cloud_id:
            cloud_id = resolve_cloud_id_from_site_basic(email, api_token, site_url)
        api_root = f"https://api.atlassian.com/ex/jira/{cloud_id}"
        return JiraConnection(
            api_root=api_root,
            auth_header=auth_basic_header(email, api_token),
            mode="scoped",
        )

    use_basic_on_site = raw_auth not in ("oauth", "bearer", "3lo")
    if raw_auth in ("oauth", "bearer", "3lo"):
        if looks_like_jira_personal_api_token(token_candidate):
            if not email_present:
                print(
                    "JIRA_AUTH_TYPE=oauth but token looks like a personal Jira API token. "
                    "Set JIRA_EMAIL for Basic auth, or use a real OAuth access token.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            use_basic_on_site = True
        elif not token_candidate:
            print("OAuth mode requires JIRA_ACCESS_TOKEN (or JIRA_API_TOKEN).", file=sys.stderr)
            raise SystemExit(1)

    if use_basic_on_site:
        base = (os.environ.get("JIRA_BASE_URL") or "").strip().strip('"').strip("'")
        email = (os.environ.get("JIRA_EMAIL") or "").strip().strip('"').strip("'")
        api_token = (
            (os.environ.get("JIRA_API_TOKEN") or os.environ.get("JIRA_ACCESS_TOKEN") or "")
            .strip()
            .strip('"')
            .strip("'")
        )
        if not base or not email or not api_token:
            print("Basic auth requires JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN (or JIRA_ACCESS_TOKEN).", file=sys.stderr)
            raise SystemExit(1)
        return JiraConnection(
            api_root=base.rstrip("/"),
            auth_header=auth_basic_header(email, api_token),
            mode="basic",
        )

    access_token = token_candidate
    site_url = (os.environ.get("JIRA_BASE_URL") or "").strip().strip('"').strip("'")
    if not site_url:
        print("Missing JIRA_BASE_URL for OAuth.", file=sys.stderr)
        raise SystemExit(1)
    cloud_id = (os.environ.get("JIRA_CLOUD_ID") or "").strip().strip('"').strip("'")
    if not cloud_id:
        cloud_id = resolve_cloud_id_from_site(access_token, site_url)
    api_root = f"https://api.atlassian.com/ex/jira/{cloud_id}"
    return JiraConnection(
        api_root=api_root,
        auth_header=f"Bearer {access_token}",
        mode="oauth",
    )


def verify_connection(conn: JiraConnection, *, quiet: bool = False) -> None:
    if conn.mode in ("basic", "scoped"):
        email = (os.environ.get("JIRA_EMAIL") or "").strip().strip('"').strip("'")
        api_token = (
            (os.environ.get("JIRA_API_TOKEN") or os.environ.get("JIRA_ACCESS_TOKEN") or "")
            .strip()
            .strip('"')
            .strip("'")
        )
        url = conn.api_root.rstrip("/") + "/rest/api/3/myself"
        headers = {"Accept": "application/json", "Authorization": conn.auth_header}
        try:
            _, me = http_json("GET", url, headers, timeout=60)
        except HttpJiraError as e:
            if conn.mode == "scoped" and e.code in (401, 403):
                print(
                    "GET /rest/api/3/myself failed with scoped token (often missing **read:jira-user**). "
                    "Search can still work with **read:jira-work** only. Continuing.",
                    file=sys.stderr,
                )
                if not quiet:
                    print_jira_api_error(e.code, e.reason, e.body)
                return
            print("GET /rest/api/3/myself failed (Basic auth).", file=sys.stderr)
            print_jira_api_error(e.code, e.reason, e.body)
            raise SystemExit(1) from e
        if not quiet and isinstance(me, dict):
            label = "scoped (gateway)" if conn.mode == "scoped" else "basic"
            print(
                f"Auth OK ({label}):",
                me.get("displayName") or me.get("accountId", ""),
                f"<{me.get('emailAddress', '')}>",
            )
        return

    token = conn.auth_header[7:].strip() if conn.auth_header.lower().startswith("bearer ") else ""
    bearer = {"Accept": "application/json", "Authorization": conn.auth_header}
    jira_me = conn.api_root.rstrip("/") + "/rest/api/3/myself"
    jira_err: HttpJiraError | None = None
    try:
        _, me = http_json("GET", jira_me, bearer, timeout=60)
        if isinstance(me, dict) and me.get("accountId"):
            if not quiet:
                print(
                    "Auth OK (oauth):",
                    me.get("displayName") or me.get("accountId", ""),
                    f"<{me.get('emailAddress', '')}>",
                )
            return
    except HttpJiraError as e:
        jira_err = e
    try:
        _, me_a = http_json("GET", ATLASSIAN_ME_URL, bearer, timeout=60)
    except HttpJiraError as e_a:
        print("OAuth token rejected (Jira /myself and Atlassian /me both failed).", file=sys.stderr)
        if e_a.code == 401:
            print_oauth_bearer_rejected_help(
                token,
                "HTTP 401 on Atlassian /me — invalid OAuth access token for Bearer auth.",
            )
        if jira_err is not None:
            print("Jira /myself:", file=sys.stderr)
            print_jira_api_error(jira_err.code, jira_err.reason, jira_err.body)
        print("Atlassian /me:", file=sys.stderr)
        print_jira_api_error(e_a.code, e_a.reason, e_a.body)
        raise SystemExit(1) from e_a
    if not quiet and isinstance(me_a, dict):
        extra = ""
        if jira_err is not None:
            extra = f" (Jira /myself HTTP {jira_err.code}; classic **read:jira-user** covers Jira profile)"
        print(
            "Auth OK (oauth): Atlassian /me —",
            me_a.get("name") or me_a.get("account_id", ""),
            extra,
        )


def search_jql_get(
    conn: JiraConnection,
    jql: str,
    max_results: int,
    next_page_token: str | None,
) -> dict:
    params: list[tuple[str, str]] = [
        ("jql", jql),
        ("maxResults", str(max_results)),
    ]
    for field in ("summary", "description", "comment"):
        params.append(("fields", field))
    if next_page_token:
        params.append(("nextPageToken", next_page_token))
    query = urlencode(params)
    url = conn.api_root.rstrip("/") + SEARCH_REL + "?" + query
    headers = {"Accept": "application/json", "Authorization": conn.auth_header}
    try:
        _, data = http_json("GET", url, headers, timeout=120)
    except HttpJiraError as e:
        if e.code == 401 and conn.auth_header.lower().startswith("bearer "):
            print_oauth_bearer_rejected_help(
                conn.auth_header[7:].strip(),
                "HTTP 401 on Jira search.",
            )
        print_oauth_scope_hint_on_forbidden(e.code, e.body)
        print_jira_api_error(e.code, e.reason, e.body)
        raise SystemExit(1) from e
    if not isinstance(data, dict):
        print("Unexpected search response (not a JSON object).", file=sys.stderr)
        raise SystemExit(1)
    return data


def search_jql_post(
    conn: JiraConnection,
    jql: str,
    max_results: int,
    next_page_token: str | None,
) -> dict:
    """POST /rest/api/3/search/jql — use when GET query string exceeds URL limits."""
    payload: dict = {
        "jql": jql,
        "maxResults": max_results,
        "fields": ["summary", "description", "comment"],
    }
    if next_page_token:
        payload["nextPageToken"] = next_page_token
    body = json.dumps(payload).encode("utf-8")
    url = conn.api_root.rstrip("/") + SEARCH_REL
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": conn.auth_header,
    }
    try:
        _, data = http_json("POST", url, headers, body=body, timeout=120)
    except HttpJiraError as e:
        print_oauth_scope_hint_on_forbidden(e.code, e.body)
        print_jira_api_error(e.code, e.reason, e.body)
        raise SystemExit(1) from e
    if not isinstance(data, dict):
        raise SystemExit(1)
    return data
