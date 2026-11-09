# test_pipeline_and_fixes.py

"""
Regression tests for the bugs fixed in the code review PR.
Run from backend/:  pytest test_pipeline_and_fixes.py
"""

import pytest

import json

import collect_docs
import process_docs
from collect_docs import Robots, _get_allowed, _read_capped, categorize, parse_page, pdf_title
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


def test_pdf_links_only_from_allowed_hosts_and_normalized():
    html = b"""<html><body><nav><a href="/documents/9/Menu.pdf">nav</a></nav><main>
        <a href="https://api.sf.gov/documents/1/Fees_2026.pdf">fees</a>
        <a href="/documents/2/Form.pdf#page=2">form</a>
        <a href="https://sf.gov/documents/2/Form.pdf?download=1">same form, other spelling</a>
        <a href="https://example.com/other.pdf">elsewhere</a>
        <a href="/information--page">not a pdf</a>
        <p>Enough text to be a page.</p>
    </main><footer><a href="/documents/8/Footer.pdf">footer</a></footer></body></html>"""
    assert parse_page(html, "https://www.sf.gov/resource--x")["pdf_links"] == [
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
    # 7-character name parts with digits are real, not random suffixes
    assert pdf_title("https://x/Plan_Phase2A.pdf") == "Plan Phase2A"
    assert pdf_title("https://x/Sheet_Rev2024.pdf") == "Sheet Rev2024"
    assert pdf_title("https://x/Form_1234567.pdf") == "Form 1234567"
    assert pdf_title("https://x/BldgForm1-2_Fillable_l3tFiXs.pdf") == "BldgForm1-2 Fillable"


class _FakeResponse:
    def __init__(self, status_code, text="", headers=None, body=b""):
        self.status_code, self.text = status_code, text
        self.headers = headers or {}
        self.body = body

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
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


def test_robots_server_error_stops_crawl_and_is_not_retried():
    session = _FakeSession({"https://www.sf.gov/robots.txt": _FakeResponse(503)})
    robots = Robots(session)
    for page in ("a", "b", "c"):
        with pytest.raises(RuntimeError):
            robots.allows(f"https://www.sf.gov/information--{page}")
    assert session.requested == ["https://www.sf.gov/robots.txt"]  # asked once, not 3 times


def test_redirect_outside_allowed_hosts_is_refused(monkeypatch):
    monkeypatch.setattr(collect_docs.time, "sleep", lambda s: None)
    session = _FakeSession({
        "https://www.sf.gov/robots.txt": _FakeResponse(200, ""),
        "https://www.sf.gov/documents/1/a.pdf":
            _FakeResponse(302, headers={"Location": "https://cdn.example.com/a.pdf"}),
    })
    with pytest.raises(RuntimeError, match="outside allowed hosts"):
        _get_allowed(session, Robots(session), "https://www.sf.gov/documents/1/a.pdf", {})
    assert "https://cdn.example.com/a.pdf" not in session.requested


def test_read_capped_stops_at_limit():
    assert _read_capped(_FakeResponse(200, body=b"x" * 10), 100) == b"x" * 10
    assert _read_capped(_FakeResponse(200, body=b"x" * 101), 100) is None
    # Declared size over the limit: rejected without reading the body
    assert _read_capped(_FakeResponse(200, headers={"Content-Length": "500"}), 100) is None


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


def test_long_header_is_shortened_instead_of_shredding_body():
    # Merged small PDF pages produce headings like "Page 1; Page 2; ...".
    heading = "; ".join(f"Page {i}" for i in range(1, 80))
    page = {
        "url": "https://www.sf.gov/documents/long.pdf",
        "title": "A long PDF",
        "category": "permits",
        "fetched_at": "2026-11-09T00:00:00+00:00",
        "sections": [{"heading": heading,
                      "paragraphs": ["The permit fee is due when you apply. " * 10]}],
    }
    chunks = chunk_page(page, "long-pdf")

    assert len(chunks) == 1  # not one chunk per word
    assert chunks[0]["text"].split("\n\n")[0].endswith("…")
    assert _tokens(chunks[0]["text"]) <= MAX_TOKENS


def test_same_path_on_two_hosts_gets_distinct_chunk_ids(tmp_path, monkeypatch):
    pages = tmp_path / "pages"
    pages.mkdir()
    for host in ("www.sf.gov", "media.api.sf.gov"):
        url = f"https://{host}/documents/form.pdf"
        page = {"url": url, "title": "Form", "category": "permits",
                "fetched_at": "2026-11-09T00:00:00+00:00", "doc_type": "pdf",
                "sections": [{"heading": "", "paragraphs": ["Fill in every field. " * 10]}]}
        (pages / collect_docs.url_to_filename(url)).write_text(json.dumps(page))
    monkeypatch.setattr(process_docs, "PAGES_DIR", pages)
    monkeypatch.setattr(process_docs, "PROCESSED_FILE", tmp_path / "chunks.json")

    assert process_docs.process_all_documents() == 0  # used to refuse: duplicate IDs
    chunks = json.loads((tmp_path / "chunks.json").read_text())
    assert len({c["id"] for c in chunks}) == len(chunks) == 2


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
