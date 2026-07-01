"""Fetch real stale/fresh page text via Wayback Machine, with LLM fallback.

- fetch_fresh(url): current live page (or most recent Wayback snapshot).
- fetch_stale(url, target_year): Wayback snapshot near target_year.
- synthesize_stale(...): GPT-5.4 fallback when no snapshot exists.

Proxy support: if HTTP_PROXY env var is set, all requests use it.
"""
import os
import re
from typing import Callable, Optional

import requests

WAYBACK_API = "https://archive.org/wayback/available"


def _get_proxy_config():
    """Return proxy dict if HTTP_PROXY is set, else None."""
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def strip_html(html: str) -> str:
    """Crude HTML → text: drop script/style, strip tags, collapse whitespace."""
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _http_get(url: str, timeout: int = 30) -> str:
    proxies = _get_proxy_config()
    resp = requests.get(url, timeout=timeout, proxies=proxies)
    resp.raise_for_status()
    return resp.text


def find_closest_snapshot(
    url: str,
    target_year: int,
    availability_fetcher: Optional[Callable[[str], dict]] = None,
) -> Optional[dict]:
    """Query Wayback availability API for the closest snapshot to Jan 1 of target_year."""
    timestamp = f"{target_year}0101"
    api_url = f"{WAYBACK_API}?url={url}&timestamp={timestamp}"
    if availability_fetcher:
        fetcher = availability_fetcher
    else:
        def fetcher(u):
            proxies = _get_proxy_config()
            return requests.get(u, timeout=20, proxies=proxies).json()
    try:
        data = fetcher(api_url)
    except Exception:
        return None
    closest = data.get("archived_snapshots", {}).get("closest", {})
    if closest.get("available"):
        return closest
    return None


def fetch_fresh(url: str, fetcher: Optional[Callable[[str], str]] = None) -> dict:
    """Fetch current page text. Fallback to most recent Wayback snapshot.

    Returns {"text": str, "timestamp": str, "age_days": 0}.
    """
    fetch = fetcher or _http_get
    try:
        html = fetch(url)
        return {"text": strip_html(html), "timestamp": "", "age_days": 0}
    except Exception:
        snap = find_closest_snapshot(url, 2026)
        if snap is None:
            return {"text": "", "timestamp": "", "age_days": 0}
        try:
            html = fetch(snap["url"])
            return {"text": strip_html(html), "timestamp": snap.get("timestamp", ""), "age_days": 0}
        except Exception:
            return {"text": "", "timestamp": "", "age_days": 0}


def fetch_stale(
    url: str,
    target_year: int,
    fetcher: Optional[Callable[[str], str]] = None,
) -> dict:
    """Fetch Wayback snapshot near target_year. Returns {"text","timestamp","age_days"} or empty."""
    snap = find_closest_snapshot(url, target_year)
    if snap is None:
        return {"text": "", "timestamp": "", "age_days": 0, "synthetic": False}
    fetch = fetcher or _http_get
    try:
        html = fetch(snap["url"])
        age_days = max(0, (2026 - target_year) * 365)
        return {"text": strip_html(html), "timestamp": snap.get("timestamp", ""), "age_days": age_days, "synthetic": False}
    except Exception:
        return {"text": "", "timestamp": "", "age_days": 0, "synthetic": False}


def synthesize_stale(
    fresh_text: str,
    gold_answer: str,
    target_year: int,
    llm_client: Optional[Callable] = None,
) -> str:
    """LLM fallback: rewrite fresh_text as it would have appeared in target_year."""
    if llm_client is None:
        return fresh_text  # nothing we can do offline
    prompt = (
        f"Rewrite the following page text as it would plausibly have appeared in {target_year}, "
        f"so that the fact '{gold_answer}' was current at that time. Keep it concise (under 300 words). "
        f"Only output the rewritten text.\n\nPage:\n{fresh_text[:1500]}"
    )
    try:
        return llm_client.chat(prompt=prompt).strip()
    except Exception:
        return fresh_text
