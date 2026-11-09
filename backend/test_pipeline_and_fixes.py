# test_pipeline_and_fixes.py

"""
Regression tests for the bugs fixed in the code review PR.
Run from backend/:  pytest test_pipeline_and_fixes.py
"""

import pytest

from collect_docs import Robots, categorize, extract_pdf_links, parse_page, pdf_title
from process_docs import MAX_TOKENS, MAX_WORDS, _tokens, chunk_page
from evaluation import EvaluationService
from api.cost_tracker import CostTracker
from api.logging_config import request_context, set_request_context, clear_request_context


# ============================================================================
# collect_docs.py
# ============================================================================

def test_categorize_selects_topic_pages():
    assert categorize("https://www.sf.gov/step-by-step--get-building-permit-house-review") == "permits"
    assert categorize("https://www.sf.gov/information--business-registration") == "business"


def test_categorize_excludes_noise():
    # Housing unit listings used to be tagged public_works via "street"
    assert categorize("https://www.sf.gov/step-by-step--1400-mission-street-unit-314") is None
    assert categorize("https://www.sf.gov/information--family-planning-services") is None
    assert categorize("https://www.sf.gov/meeting--building-inspection-commission") is None
    assert categorize("https://example.com/information--building-permits") is None


def test_parse_page_keeps_symbols_and_headings():
    html = b"""<html><body><nav>Menu</nav><main>
        <h1>Parking permits</h1><p>Intro text.</p>
        <h2>Cost</h2><p>The permit costs $170 per year. See sf.gov/parking or email a@sfmta.com.</p>
        <h3>Discounts</h3><ul><li>Low income: 50% off</li></ul>
    </main><footer>Footer</footer></body></html>"""
    page = parse_page(html)

    assert page["title"] == "Parking permits"
    headings = [s["heading"] for s in page["sections"]]
    assert headings == ["", "Cost", "Cost › Discounts"]
    text = " ".join(p for s in page["sections"] for p in s["paragraphs"])
    assert "$170" in text and "sf.gov/parking" in text and "a@sfmta.com" in text and "50%" in text
    assert "Menu" not in text and "Footer" not in text


def test_extract_pdf_links_only_from_allowed_hosts():
    html = b"""<main>
        <a href="https://api.sf.gov/documents/1/Fees_2026.pdf">fees</a>
        <a href="/documents/2/Form.pdf#page=2">form</a>
        <a href="https://example.com/other.pdf">elsewhere</a>
        <a href="/information--page">not a pdf</a>
    </main>"""
    assert extract_pdf_links(html, "https://www.sf.gov/resource--x") == [
        "https://api.sf.gov/documents/1/Fees_2026.pdf",
        "https://www.sf.gov/documents/2/Form.pdf",
    ]


def test_pdf_title_cleans_file_name():
    assert pdf_title("https://media.api.sf.gov/documents/Table_1A-A_-_Building_Permit_Fees_2026.pdf") \
        == "Table 1A-A - Building Permit Fees 2026"
    # random suffix removed, real words kept
    assert pdf_title("https://x/documents/Property_Owner_Disclosure_Form_589lGbh.pdf") \
        == "Property Owner Disclosure Form"
    assert pdf_title("https://x/documents/Electrical_Permits.pdf") == "Electrical Permits"


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code, self.text = status_code, text


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, timeout):
        return self.responses[url]


def test_robots_rules_and_missing_robots_file():
    robots = Robots(_FakeSession({
        "https://api.sf.gov/robots.txt": _FakeResponse(200, "User-agent: *\nDisallow: /"),
        "https://media.api.sf.gov/robots.txt": _FakeResponse(403),  # no robots.txt (RFC 9309)
    }))
    assert robots.allows("https://api.sf.gov/documents/1/a.pdf") is False
    assert robots.allows("https://media.api.sf.gov/documents/a.pdf") is True


def test_disallowed_pdf_is_skipped_without_waiting(monkeypatch):
    import collect_docs
    sleeps = []
    monkeypatch.setattr(collect_docs.time, "sleep", lambda s: sleeps.append(s))
    robots = Robots(_FakeSession({
        "https://api.sf.gov/robots.txt": _FakeResponse(200, "User-agent: *\nDisallow: /"),
    }))
    session = _FakeSession({})  # any real request would raise KeyError
    assert collect_docs._get_allowed(session, robots, "https://api.sf.gov/documents/1/a.pdf", {}) is None
    assert sleeps == []


def test_robots_server_error_stops_crawl():
    robots = Robots(_FakeSession({"https://www.sf.gov/robots.txt": _FakeResponse(503)}))
    with pytest.raises(RuntimeError):
        robots.allows("https://www.sf.gov/information--x")


# ============================================================================
# process_docs.py
# ============================================================================

def test_chunks_have_header_metadata_and_fit_word_cap():
    long_paragraph = " ".join(f"word{i}." for i in range(500))
    page = {
        "url": "https://www.sf.gov/information--example",
        "title": "Example page",
        "category": "permits",
        "fetched_at": "2026-11-09T00:00:00+00:00",
        "sections": [{"heading": "Fees", "paragraphs": ["Fee is $76.", long_paragraph]}],
    }
    chunks = chunk_page(page, "information-example")

    assert len(chunks) > 1
    for chunk in chunks:
        header, body = chunk["text"].split("\n\n", 1)
        assert header == "Example page › Fees"
        assert len(body.split()) <= MAX_WORDS
        assert chunk["metadata"]["url"] == page["url"]
    assert "$76" in chunks[0]["text"]


def test_dense_text_is_split_to_fit_token_limit():
    # ~150 "words" but each is a dense code like a PDF table cell: far more
    # than 256 tokens. Word-based packing alone let these through.
    dense_lines = [" ".join(f"A{i}-{j}/B{j}.{i}" for j in range(10)) for i in range(15)]
    page = {
        "url": "https://www.sf.gov/documents/table.pdf",
        "title": "Fee table",
        "category": "permits",
        "fetched_at": "2026-11-09T00:00:00+00:00",
        "sections": [{"heading": "Page 1", "paragraphs": dense_lines}],
    }
    chunks = chunk_page(page, "fee-table")

    assert len(chunks) > 1
    assert all(_tokens(chunk["text"]) <= MAX_TOKENS for chunk in chunks)


# ============================================================================
# evaluation.py
# ============================================================================

def _evaluator():
    return EvaluationService.__new__(EvaluationService)


def test_hallucination_check_accepts_facts_found_in_sources():
    # Used to read source['content'], which never existed, so every
    # phone number / address counted as a hallucination.
    contexts = ["Call 628-652-3200 or visit 100 Larkin Street."]
    answer = "Call 628-652-3200 or visit 100 Larkin Street."
    assert _evaluator()._check_hallucination_indicators(answer, contexts) == 0.0


def test_hallucination_check_flags_facts_missing_from_sources():
    contexts = ["Call 628-652-3200."]
    assert _evaluator()._check_hallucination_indicators("Call 415-555-1234.", contexts) == 0.5


# ============================================================================
# api/cost_tracker.py
# ============================================================================

def test_haiku_is_priced_at_haiku_rates():
    tracker = CostTracker.__new__(CostTracker)
    cost = tracker.calculate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
    assert cost["input_cost"] == 1.0 and cost["output_cost"] == 5.0


# ============================================================================
# api/logging_config.py
# ============================================================================

def test_set_request_context_keeps_earlier_fields():
    try:
        set_request_context(request_id="req-1", ip_address="127.0.0.1")
        set_request_context(conversation_id="conv-1")
        context = request_context.get()
        assert context["request_id"] == "req-1"
        assert context["conversation_id"] == "conv-1"
    finally:
        clear_request_context()
