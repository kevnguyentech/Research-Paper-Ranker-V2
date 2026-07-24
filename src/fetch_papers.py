import json
import re
import time

import requests

from src.config import (
    BASE_URL,
    CACHE_DIR,
    CACHE_TTL_DAYS,
    PAPER_FIELDS,
    REQUEST_DELAY,
    S2_API_KEY,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cache_path(key: str):
    safe = re.sub(r"[^\w\-]", "_", key)[:120]
    return CACHE_DIR / f"{safe}.json"


def _cache_valid(path) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days < CACHE_TTL_DAYS


def _get(url: str, params: dict) -> dict:
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    delay = 0.15 if S2_API_KEY else REQUEST_DELAY

    for attempt in range(5):
        try:
            time.sleep(delay)
            resp = requests.get(url, params=params, headers=headers, timeout=15)
        except requests.exceptions.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"S2 API: network error after 5 attempts: {exc}") from exc
            print(f"Network error (attempt {attempt + 1}/5): {exc}. Retrying...")
            continue
        if resp.status_code == 429:
            try:
                wait = int(resp.headers.get("Retry-After", 5)) + attempt * 2
            except (ValueError, TypeError):
                wait = 5 + attempt * 2
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            if attempt == 4:
                resp.raise_for_status()
            print(f"Server error {resp.status_code} (attempt {attempt + 1}/5). Retrying...")
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("S2 API: too many retries")


# ── public functions ──────────────────────────────────────────────────────────

def fetch_by_topic(query: str, limit: int = 100, refresh: bool = False) -> list[dict]:
    """Search papers by topic or research question."""
    cache_key = "topic_" + re.sub(r"\W+", "_", query.lower())[:60] + f"_n{limit}"
    cache = _cache_path(cache_key)

    if not refresh and _cache_valid(cache):
        return json.loads(cache.read_text())

    params = {
        "query": query,
        "fields": PAPER_FIELDS,
        "limit": min(limit, 100),
    }
    data = _get(f"{BASE_URL}/paper/search", params)
    papers = [p for p in data.get("data", []) if p.get("abstract")]
    cache.write_text(json.dumps(papers))
    return papers


def resolve_paper(query: str, refresh: bool = False) -> str | None:
    if re.fullmatch(r"[0-9a-f]{40}", query.strip()):
        return query.strip()

    cache = _cache_path("resolve_" + re.sub(r"\W+", "_", query.lower())[:60])
    if not refresh and _cache_valid(cache):
        return json.loads(cache.read_text()).get("paperId")

    params = {"query": query, "fields": "paperId,title", "limit": 1}
    data = _get(f"{BASE_URL}/paper/search", params)
    hits = data.get("data", [])
    if not hits:
        return None
    print(f"Resolved '{query}' -> '{hits[0]['title']}'")
    cache.write_text(json.dumps(hits[0]))
    return hits[0]["paperId"]


def fetch_citations(
    paper_id: str,
    direction: str = "both",
    limit: int = 50,
    refresh: bool = False,
) -> list[dict]:
    """
    Fetch papers connected via the citation graph.
      forward  = papers that CITE this paper (newer work)
      backward = papers this paper CITES (foundational sources)
      both     = union, deduplicated by paperId
    """
    endpoints = {
        "forward":  ("citations",  "citingPaper"),
        "backward": ("references", "citedPaper"),
    }

    results: dict[str, dict] = {}

    for dir_label, (endpoint, wrapper_key) in endpoints.items():
        if direction not in (dir_label, "both"):
            continue

        cache = _cache_path(f"{dir_label}_{paper_id}_n{limit}")

        if not refresh and _cache_valid(cache):
            for p in json.loads(cache.read_text()):
                results[p["paperId"]] = p
            continue

        params = {"fields": PAPER_FIELDS, "limit": min(limit, 100)}
        data = _get(f"{BASE_URL}/paper/{paper_id}/{endpoint}", params)

        papers = [
            item[wrapper_key]
            for item in data.get("data", [])
            if item.get(wrapper_key) and item[wrapper_key].get("paperId")
            and item[wrapper_key].get("abstract")
        ]
        cache.write_text(json.dumps(papers))
        for p in papers:
            results[p["paperId"]] = p

    return list(results.values())