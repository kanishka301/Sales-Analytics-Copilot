"""
Optional web search for EXTERNAL context only (holidays, news, events)."""
import os


def web_search(query: str, max_results: int = 3) -> dict:
    max_results = max(1, min(int(max_results), 5))
    results = []
    key = os.getenv("TAVILY_API_KEY")
    if key:
        from tavily import TavilyClient
        r = TavilyClient(api_key=key).search(query, max_results=max_results)
        results = [{"title": x.get("title"), "url": x.get("url"), "snippet": (x.get("content") or "")[:300]}
                   for x in r.get("results", [])]
        source = "tavily"
    else:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as d:
            raw = d.text(query, max_results=max_results)
        results = [{"title": x.get("title"), "url": x.get("href"), "snippet": (x.get("body") or "")[:300]}
                   for x in raw]
        source = "duckduckgo"
    return {
        "source": source, "results": results,
        "warning": "EXTERNAL and unverified. Use only as possible context, never as sales data.",
    }
