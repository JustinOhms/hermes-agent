"""Self-hosted web research via SearXNG + simple HTML fetcher.

Two tools registered in the 'web' toolset:
  - searxng_search  : query a local SearXNG instance (Google/Bing/DDG/Brave aggregator)
  - web_fetch_clean : fetch a URL and return cleaned text (Firecrawl-lite, no deps)

Both activate whenever SEARXNG_URL is set in ~/.hermes/.env.
"""
import html
import json
import os
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from tools.registry import registry


# --------------------------------------------------------------------------- #
# SearXNG search                                                              #
# --------------------------------------------------------------------------- #

def _searxng_url() -> str:
    return os.getenv("SEARXNG_URL", "").strip().rstrip("/")


def check_searxng() -> bool:
    return bool(_searxng_url())


def searxng_search(query: str, num_results: int = 10, categories: str = "general",
                   task_id: str = None) -> str:
    base = _searxng_url()
    if not base:
        return json.dumps({"success": False, "error": "SEARXNG_URL is not set in .env"})

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "safesearch": os.getenv("SEARXNG_SAFESEARCH", "0"),
        "language": os.getenv("SEARXNG_LANGUAGE", "en"),
    }
    url = f"{base}/search?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "hermes-agent/searxng"})

    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as e:
        return json.dumps({"success": False, "error": f"SearXNG HTTP {e.code}: {e.reason}"})
    except URLError as e:
        return json.dumps({"success": False, "error": f"SearXNG unreachable at {base}: {e.reason}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"SearXNG error: {e}"})

    raw = data.get("results") or []
    results = []
    for r in raw[: max(1, min(int(num_results), 25))]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:400],
            "engine": r.get("engine", ""),
            "published": r.get("publishedDate"),
        })

    return json.dumps({
        "success": True,
        "query": data.get("query", query),
        "num_returned": len(results),
        "results": results,
        "answers": data.get("answers", []),
        "infoboxes": [{"content": ib.get("content", "")[:400]}
                      for ib in (data.get("infoboxes") or [])],
    }, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# web_fetch_clean — dependency-free HTML → text                               #
# --------------------------------------------------------------------------- #

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>",
                              re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _html_to_text(raw: str) -> tuple[str, str]:
    """Return (title, text) from raw HTML. Simple, deps-free."""
    title_m = _TITLE_RE.search(raw)
    title = html.unescape(title_m.group(1).strip()) if title_m else ""

    # Strip scripts/styles entirely
    cleaned = _SCRIPT_STYLE_RE.sub(" ", raw)
    # Replace block tags with newlines before stripping all tags
    cleaned = re.sub(r"</?(p|div|br|li|tr|h[1-6]|article|section|header|footer)[^>]*>",
                     "\n", cleaned, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = _TAG_RE.sub("", cleaned)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return title, text.strip()


def web_fetch_clean(url: str, max_chars: int = 8000, task_id: str = None) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return json.dumps({"success": False, "error": "url must start with http:// or https://"})

    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; hermes-agent/web_fetch_clean)",
        "Accept": "text/html,application/xhtml+xml",
    })
    try:
        with urlopen(req, timeout=20) as resp:
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in ctype:
                charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
            raw = resp.read(2_000_000)  # 2 MB cap
    except HTTPError as e:
        return json.dumps({"success": False, "error": f"HTTP {e.code}: {e.reason}", "url": url})
    except URLError as e:
        return json.dumps({"success": False, "error": f"Unreachable: {e.reason}", "url": url})
    except Exception as e:
        return json.dumps({"success": False, "error": f"fetch error: {e}", "url": url})

    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")

    title, text = _html_to_text(body)
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return json.dumps({
        "success": True,
        "url": url,
        "title": title,
        "num_chars": len(text),
        "truncated": truncated,
        "content": text,
    }, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #

registry.register(
    name="searxng_search",
    toolset="web",
    schema={
        "name": "searxng_search",
        "description": (
            "Search the web via a self-hosted SearXNG instance. Returns titles, URLs, "
            "and snippets from aggregated search engines (Google, Bing, DuckDuckGo, "
            "Brave, etc.). PREFERRED search tool — free, private, no rate limits. "
            "Use this instead of web_search unless it's unavailable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "num_results": {"type": "integer", "description": "Max results (1-25, default 10)", "default": 10},
                "categories": {
                    "type": "string",
                    "description": "SearXNG category: general, news, images, videos, it, science, files, map, music, social media",
                    "default": "general",
                },
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: searxng_search(
        query=args.get("query", ""),
        num_results=args.get("num_results", 10),
        categories=args.get("categories", "general"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_searxng,
    requires_env=["SEARXNG_URL"],
)

registry.register(
    name="web_fetch_clean",
    toolset="web",
    schema={
        "name": "web_fetch_clean",
        "description": (
            "Fetch a URL and return cleaned plain text (Firecrawl-lite). Strips scripts, "
            "styles, and HTML tags. Use this to read articles, docs, or any page after "
            "searxng_search returns URLs. Pair with searxng_search for full research flow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http/https URL"},
                "max_chars": {"type": "integer", "description": "Max characters of body text to return (default 8000)", "default": 8000},
            },
            "required": ["url"],
        },
    },
    handler=lambda args, **kw: web_fetch_clean(
        url=args.get("url", ""),
        max_chars=args.get("max_chars", 8000),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_searxng,   # same gate — enabled alongside searxng_search
    requires_env=["SEARXNG_URL"],
)
