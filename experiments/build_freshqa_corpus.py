#!/usr/bin/env python3
"""Build FreshQA stale/fresh document corpus from Wayback Machine.

For each FreshQA dev query, fetch two snapshots of the source URL:
  - "stale": earliest snapshot from the past 2+ years
  - "fresh": latest snapshot from the past 12 months

Writes data/freshqa_corpus/<id>_{stale,fresh}.txt (plain text extracted
from the HTML) and <id>_meta.json (timestamps, age_days).

Used by experiments/decomp_freshqa/run_decomp.py --mode freshqa for the
direct causal ablation that Reviewer 2 asked for.
"""
import json
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "freshqa_corpus"
CORPUS_DIR.mkdir(exist_ok=True)

CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_ID = "https://web.archive.org/web/{ts}id_/{url}"

# Wayback freshness windows per doc state:
#   stale = earliest snapshot >= 18 months before "now" (matches the
#           low-drift / Wayback-aged FreshQA setup)
#   fresh = latest snapshot <= 6 months before "now"
STALE_MIN_AGE_MONTHS = 18
FRESH_MAX_AGE_MONTHS = 6

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 FreshQA-Build"


class TextExtractor(HTMLParser):
    """Strip tags, return visible text. Suitable for Wikipedia-style pages."""
    SKIP = {"script", "style", "noscript", "head", "nav"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if self.skip == 0 and data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        out = "".join(self.parts)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n\s*\n+", "\n\n", out)
        return out.strip()


def html_to_text(html: str) -> str:
    p = TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return p.get_text()


def cdx_snapshots(url: str) -> list:
    """Return [(timestamp_str, mimetype, statuscode)] for HTTP 200 text/html snapshots of url."""
    params = {
        "url": url,
        "output": "json",
        "limit": 200,
        "filter": "statuscode:200",
    }
    for attempt in range(5):
        try:
            r = requests.get(CDX_URL, params=params, timeout=90,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                rows = r.json()
                if len(rows) < 2:
                    return []
                header = rows[0]
                out = []
                for row in rows[1:]:
                    rec = dict(zip(header, row))
                    if "text/html" in (rec.get("mimetype") or ""):
                        out.append(rec["timestamp"])
                return out
        except Exception as e:
            print(f"     cdx retry {attempt+1}/5: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def pick_snapshots(timestamps: list) -> tuple:
    """Return (stale_ts, fresh_ts) given a sorted-ascending list of YYYYMMDDhhmmss strings."""
    if not timestamps:
        return None, None
    fresh = timestamps[-1]            # most recent
    n = len(timestamps)
    # Stale = pick one at index floor(n*0.25) so it's clearly older than the
    # median but still a successful snapshot — not the absolute oldest (which
    # might be from when the page was a stub).
    stale_idx = max(0, n // 4)
    stale = timestamps[stale_idx]
    if stale == fresh and n >= 2:
        stale = timestamps[0]
    return stale, fresh


def fetch_snapshot(url: str, ts: str, retries: int = 3) -> str:
    """Fetch raw HTML for a Wayback timestamp."""
    full = WAYBACK_ID.format(ts=ts, url=url)
    for attempt in range(retries):
        try:
            r = requests.get(full, timeout=45, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except Exception as e:
            print(f"     retry {attempt+1}/{retries}: {type(e).__name__}")
            time.sleep(2)
    return ""


def build_one(qid: str, url: str) -> bool:
    """Build stale/fresh doc files for one query; return True on success."""
    stale_path = CORPUS_DIR / f"{qid}_stale.txt"
    fresh_path = CORPUS_DIR / f"{qid}_fresh.txt"
    meta_path = CORPUS_DIR / f"{qid}_meta.json"
    if stale_path.exists() and fresh_path.exists() and meta_path.exists():
        return True

    timestamps = cdx_snapshots(url)
    if not timestamps:
        return False
    stale_ts, fresh_ts = pick_snapshots(timestamps)
    if not stale_ts or not fresh_ts:
        return False

    try:
        stale_html = fetch_snapshot(url, stale_ts)
        fresh_html = fetch_snapshot(url, fresh_ts)
    except Exception:
        return False
    if not stale_html or not fresh_html:
        return False

    stale_text = html_to_text(stale_html)[:8000]
    fresh_text = html_to_text(fresh_html)[:8000]
    if len(stale_text) < 100 or len(fresh_text) < 100:
        return False

    stale_path.write_text(stale_text, encoding="utf-8")
    fresh_path.write_text(fresh_text, encoding="utf-8")
    meta_path.write_text(json.dumps({
        "stale_timestamp": stale_ts,
        "fresh_timestamp": fresh_ts,
    }), encoding="utf-8")
    return True


def main():
    parquet_path = DATA_DIR / "freshqa_dev.parquet"
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} FreshQA dev rows")

    # Restrict to 74 dev queries used in the paper (first 74 of the dev split)
    df = df.head(74)
    n_ok = n_fail = 0
    for i, row in df.iterrows():
        qid = str(row["id"])
        url = str(row.get("source", ""))
        if not url:
            n_fail += 1
            continue
        # Wayback doesn't accept anchor-style URLs; drop the #fragment
        url_clean = url.split("#")[0]
        try:
            ok = build_one(qid, url_clean)
        except Exception as e:
            print(f"  [{i+1}] {qid}: {type(e).__name__}", flush=True)
            ok = False
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(df)}] built={n_ok} failed={n_fail}", flush=True)
        time.sleep(1.5)  # be polite to archive.org (CDX + 2 fetches = 3 requests per query)

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed out of {len(df)}")


if __name__ == "__main__":
    main()