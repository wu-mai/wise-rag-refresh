import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fresh_rag.backends.wayback import strip_html, find_closest_snapshot, synthesize_stale


def test_strip_html_removes_tags():
    html = "<html><body><p>Hello <b>world</b></p><script>x()</script></body></html>"
    text = strip_html(html)
    assert "Hello world" in text
    assert "x()" not in text


def test_strip_html_collapses_whitespace():
    html = "<p>foo\n\n   bar</p>"
    assert "  " not in strip_html(html)
    assert "foo bar" in strip_html(html)


def test_find_closest_snapshot_uses_fetcher():
    calls = []
    def fake_fetch(url):
        calls.append(url)
        return {"archived_snapshots": {"closest": {"available": True, "timestamp": "20200102", "url": "http://wb/x"}}}
    snap = find_closest_snapshot("example.com", 2020, availability_fetcher=fake_fetch)
    assert snap["timestamp"] == "20200102"
    assert calls  # fetcher was called


def test_find_closest_snapshot_none_when_unavailable():
    def fake_fetch(url):
        return {"archived_snapshots": {}}
    assert find_closest_snapshot("example.com", 2020, availability_fetcher=fake_fetch) is None


def test_synthesize_stale_uses_llm():
    class FakeLLM:
        def __init__(self): self.called = False
        def chat(self, prompt, **kw):
            self.called = True
            return "In 2020, the prime minister was X."
    llm = FakeLLM()
    text = synthesize_stale("current page text", "the PM is Y", 2020, llm_client=llm)
    assert llm.called
    assert "prime minister was X" in text
