"""
arxiv paper tools - hugging face papers api (see week_3/3_paper_tools.md)

this replaces week 2's alphaxiv mcp server. no mcp, no oauth bridge - just two
plain http calls. trade-off: HF papers only indexes ML/CS papers that made it to
huggingface.co/papers (daily papers, model card links etc), not all of arxiv.
read_paper falls back to "not indexed" so the agent knows to try web_fetch on
arxiv.org/abs/... instead.
"""

import os
import re
import requests

BASE_URL = "https://huggingface.co"
MAX_PAPER_CHARS = 15_000


def _headers() -> dict:
    # hf_token is optional, just helps with rate limits
    token = os.environ.get("hf_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def normalize_arxiv_id(raw: str) -> str:
    """
    strip url/prefix noise down to the bare id. version suffixes (v1, v2...) are
    kept as-is, not stripped - HF's own paper-id parsing table does the same
    (e.g. "2602.08025v1" stays "2602.08025v1"), so this matches their API's expectations.
    """
    s = raw.strip()
    s = re.sub(r"^arxiv:", "", s, flags=re.IGNORECASE)
    if "arxiv.org" in s or "huggingface.co" in s:
        s = s.rstrip("/").split("/")[-1]
    s = re.sub(r"\.(pdf|md)$", "", s)
    return s


def paper_search(query: str, limit: int = 5) -> dict:
    """find papers on huggingface.co/papers by keyword (hybrid semantic + full-text search)."""
    try:
        r = requests.get(
            f"{BASE_URL}/api/papers/search",
            params={"q": query},
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return {"error": f"paper_search failed: {e}"}
    except ValueError:
        return {"error": "paper_search returned non-json response"}

    papers = []
    for item in (data or [])[:limit]:
        # response shape varies - sometimes the paper is wrapped in a "paper" key, sometimes not
        p = item.get("paper", item) if isinstance(item, dict) else {}
        arxiv_id = p.get("id") or p.get("arxiv_id") or item.get("id")
        title = p.get("title") or item.get("title") or ""
        summary = p.get("summary") or p.get("abstract") or ""
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "snippet": (summary[:300] + "...") if len(summary) > 300 else summary,
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        })

    return {"query": query, "papers": papers, "count": len(papers)}


def read_paper(arxiv_id: str) -> dict:
    """metadata + markdown content for a paper. falls back to abstract if .md isn't available."""
    aid = normalize_arxiv_id(arxiv_id)

    try:
        r = requests.get(f"{BASE_URL}/api/papers/{aid}", headers=_headers(), timeout=15)
    except requests.RequestException as e:
        return {"error": f"read_paper failed: {e}"}

    if r.status_code == 404:
        return {
            "error": (
                f"'{aid}' is not indexed on hugging face papers. "
                f"try web_fetch('https://arxiv.org/abs/{aid}') instead."
            )
        }
    if not r.ok:
        return {"error": f"read_paper failed: HTTP {r.status_code}"}

    try:
        meta = r.json()
    except ValueError:
        return {"error": "read_paper got non-json metadata response"}

    title = meta.get("title", "")
    abstract = meta.get("summary") or meta.get("abstract", "")

    # .md uses arxiv html when available - not every paper has it, abstract fallback is normal
    content = abstract
    try:
        md_r = requests.get(f"{BASE_URL}/papers/{aid}.md", headers=_headers(), timeout=15)
        if md_r.ok and md_r.text.strip():
            content = md_r.text
    except requests.RequestException:
        pass

    truncated = False
    if len(content) > MAX_PAPER_CHARS:
        content = content[:MAX_PAPER_CHARS]
        truncated = True

    return {
        "arxiv_id": aid,
        "title": title,
        "abstract": abstract,
        "content": content,
        "truncated": truncated,
        "url": f"https://arxiv.org/abs/{aid}",
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "paper_search",
            "description": (
                "Search ML/CS papers indexed on huggingface.co/papers (a large subset of arxiv, "
                "not all of it). Use for academic/research questions before web_search. "
                "Returns small previews - call read_paper for full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "description": "default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Get a paper's metadata and markdown content by arxiv_id (from paper_search results - "
                "don't guess IDs). If it returns a 'not indexed' error, fall back to web_fetch on "
                "arxiv.org/abs/<id>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "e.g. 2205.14135 or a full arxiv url"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
]

DISPATCH = {
    "paper_search": paper_search,
    "read_paper": read_paper,
}
