"""
web search and fetch - carried over from week 2 (serper + trafilatura/markdownify).

week 2 wrapped these in asyncio.to_thread because the alphaxiv mcp client forced the
whole agent loop to be async. week 3 drops mcp entirely, so the agent loop is plain
sync now - these are just plain functions, no async needed.
"""

import os
import requests
import trafilatura
from markdownify import markdownify as md

UA = "Mozilla/5.0 (research-agent)"
MAX_FETCH_CHARS = 6000


def web_search(query: str, num_results: int = 5) -> dict:
    """serper/google search - titles, links, snippets."""
    key = os.environ.get("serper_api_key")
    if not key:
        return {"error": "serper_api_key not set in .env"}

    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return {"error": f"web_search failed: {e}"}

    results = []
    box = data.get("answerBox")
    if box:
        ans = box.get("answer") or box.get("snippet")
        if ans:
            results.append({"type": "answer_box", "text": ans})

    for item in data.get("organic", [])[:num_results]:
        results.append({
            "title": item.get("title"),
            "url": item.get("link"),
            "snippet": item.get("snippet", ""),
        })

    return {"query": query, "results": results}


def web_fetch(url: str, max_chars: int = MAX_FETCH_CHARS) -> dict:
    """fetch a page and return cleaned text (trafilatura first, markdownify fallback)."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"web_fetch failed: {e}"}

    text = trafilatura.extract(r.text, include_comments=False, include_tables=True)
    if not text:
        text = md(r.text, strip=["script", "style"])

    text = (text or "").strip()
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return {"url": url, "content": text or "(empty page)", "truncated": truncated}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web (Serper/Google). Returns titles, links, snippets. "
                "Use for news, blogs, docs - anything non-academic. Use paper_search "
                "instead for ML research questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "description": "default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its cleaned text. Use after web_search to read a page in full - don't fetch blindly.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

DISPATCH = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}
