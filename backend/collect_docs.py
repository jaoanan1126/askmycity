"""
Step 1: Collect SF government pages from sf.gov

How it works
------------
1. Discover pages from sf.gov's sitemap.xml (complete and deterministic —
   no link-following, no depth limits).
2. Keep only useful page types (how-to, step-by-step, resources, locations,
   departments) whose URL matches one of our TOPICS. The matching topic
   becomes the page's category.
3. Fetch each page politely (robots.txt, 1 request/second). Re-crawls send
   the previous ETag/Last-Modified, so unchanged pages come back as
   304 Not Modified and are reused instead of re-downloaded.
4. Save each page as JSON with its URL, title and heading-based sections.
   PDFs linked from those pages are fetched too (one section per PDF
   page), when robots.txt allows it on every host along the redirect chain.
5. Write everything to a fresh staging directory and only replace
   data/raw/ if the crawl looks healthy. Otherwise exit with an error and
   keep the previous data untouched.

Usage
-----
    python collect_docs.py      # full crawl, replaces data/raw on success

Output layout (data/raw/):
    manifest.json          crawl summary + per-URL ETag/hash for re-crawls
    pages/<slug>.json      one file per page
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

SITE = "https://www.sf.gov"
SITEMAP_URL = f"{SITE}/sitemap.xml"

# PDFs linked from crawled pages are fetched from these hosts only.
# robots.txt is still checked per host (api.sf.gov currently disallows all).
PDF_HOSTS = {"www.sf.gov", "sf.gov", "api.sf.gov", "media.api.sf.gov"}
MAX_PDF_BYTES = 10_000_000
MAX_REDIRECTS = 3

USER_AGENT = "AskMyCityBot/1.0 (educational RAG project)"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 20

# sf.gov URL slugs start with the page type, e.g. "information--...".
# Skipped on purpose: meeting--, news--, events--, profile--, reports--, data--
INCLUDED_PAGE_TYPES = (
    "information--",
    "step-by-step--",
    "resource--",
    "location--",
    "departments--",
)

# Topic -> keywords matched as whole words in the URL slug (plural "s" ok).
# The first matching topic becomes the page's category. Order matters.
TOPICS: dict[str, list[str]] = {
    "permits": ["permit", "building", "remodel", "inspection", "construction", "dbi", "adu"],
    "parking": ["parking", "meter", "tow", "citation", "sfmta"],
    "business": ["business", "license", "registration", "small-business", "storefront", "vendor"],
    "planning": ["sf-planning", "zoning", "variance", "land-use", "historic-preservation",
                 "conditional-use", "planning-permit", "planning-application"],
    "public_works": ["public-works", "sidewalk", "street-tree", "street-cleaning", "street-use",
                     "graffiti", "pothole", "curb", "litter", "encroachment"],
}

# Slugs matching any of these are never crawled, even if a topic matches.
EXCLUDE_PATTERNS = [re.compile(p) for p in (
    r"^[a-z-]+--\d",          # address-style listings, e.g. step-by-step--1400-mission-street-...
    r"-unit-",                # housing unit listings
    r"commission|committee",  # governance bodies, not services
    r"capacity-building|community-building|building-trust|building-future|building-transparency",
    r"family-planning|juvenile|youth-c(linic|enter)|overdose|street-(outreach|response|crises)",
    r"voter|covid|mask-use|software-license|mobile-device|disclosure-gifts",
    r"scorecard|annual-reports|rfq|nofa|recommendations-\d{4}",
)]

# Health checks for a full crawl: fail instead of replacing data/raw.
MAX_FAILURE_RATE = 0.2   # more than 20% of fetches failed -> abort
MIN_SECTION_WORDS_PER_PAGE = 30  # pages with less text are skipped as empty

# Elements that are page chrome, not content.
STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "button"]
# Block elements whose text we keep, in document order.
BLOCK_TAGS = ["p", "li", "td", "th", "dt", "dd", "address", "pre", "blockquote",
              "h4", "h5", "h6"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _keyword_pattern(keyword: str) -> re.Pattern:
    return re.compile(r"(^|-)" + re.escape(keyword) + r"s?(-|$)")


TOPIC_PATTERNS = {
    topic: [_keyword_pattern(k) for k in keywords] for topic, keywords in TOPICS.items()
}


def categorize(url: str) -> Optional[str]:
    """Return the topic for a URL, or None if it should not be crawled."""
    parsed = urlparse(url)
    if parsed.netloc not in ("www.sf.gov", "sf.gov"):
        return None
    slug = parsed.path.strip("/").lower()
    if "/" in slug or not slug.startswith(INCLUDED_PAGE_TYPES):
        return None
    if any(p.search(slug) for p in EXCLUDE_PATTERNS):
        return None
    for topic, patterns in TOPIC_PATTERNS.items():
        if any(p.search(slug) for p in patterns):
            return topic
    return None


def discover_urls(session: requests.Session) -> dict[str, str]:
    """Return {url: category} for every sitemap page we want."""
    response = session.get(SITEMAP_URL, timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_urls = [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns) if loc.text]

    selected: dict[str, str] = {}
    for url in sitemap_urls:
        category = categorize(url)
        if category:
            selected[url.rstrip("/")] = category
    return dict(sorted(selected.items()))


class Robots:
    """
    robots.txt rules per host, loaded on first use.

    Follows RFC 9309: a 4xx response (other than 429) means the site has
    no robots.txt, so nothing is disallowed. Network errors, 429 and 5xx
    raise, because we never crawl blind.
    """

    def __init__(self, session: requests.Session):
        self.session = session
        self._rules: dict[str, RobotFileParser] = {}

    def allows(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._rules:
            response = self.session.get(f"https://{host}/robots.txt",
                                        timeout=REQUEST_TIMEOUT_SECONDS)
            rules = RobotFileParser()
            if response.status_code == 200:
                rules.parse(response.text.splitlines())
            elif 400 <= response.status_code < 500 and response.status_code != 429:
                rules.parse([])
            else:
                raise RuntimeError(f"robots.txt for {host} unavailable "
                                   f"(HTTP {response.status_code})")
            self._rules[host] = rules
        return self._rules[host].can_fetch(USER_AGENT, url)


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace. Keeps all symbols ($, %, /, @, ...)."""
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _has_block_ancestor(el: Tag, root: Tag) -> bool:
    """True if el is nested in another block we already take text from."""
    for parent in el.parents:
        if parent is root:
            return False
        if parent.name in BLOCK_TAGS:
            return True
    return False


def parse_page(html: bytes) -> dict:
    """
    Extract the title and heading-based sections from an sf.gov page.

    Returns {"title": str, "sections": [{"heading": str, "paragraphs": [str]}]}
    where heading is "" for text before the first <h2>, "H2" or "H2 › H3".
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup

    h1 = main.find("h1") or soup.find("h1")
    if h1:
        title = _clean(h1.get_text(" "))
    elif soup.title:
        title = _clean(soup.title.get_text()).removesuffix("| SF.gov").strip()
    else:
        title = ""

    sections: list[dict] = [{"heading": "", "paragraphs": []}]
    h2 = ""
    for el in main.find_all(["h2", "h3", *BLOCK_TAGS]):
        if el.name == "h2":
            h2 = _clean(el.get_text(" "))
            sections.append({"heading": h2, "paragraphs": []})
        elif el.name == "h3":
            h3 = _clean(el.get_text(" "))
            sections.append({"heading": f"{h2} › {h3}" if h2 else h3, "paragraphs": []})
        elif not _has_block_ancestor(el, main):
            text = _clean(el.get_text(" "))
            if text:
                sections[-1]["paragraphs"].append(text)

    sections = [s for s in sections if s["paragraphs"]]
    return {"title": title, "sections": sections}


def extract_pdf_links(html: bytes, base_url: str) -> list[str]:
    """Absolute URLs of PDFs linked from the page's main content."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup
    links = []
    for a in main.find_all("a", href=True):
        url = urljoin(base_url, a["href"].strip()).split("#")[0]
        parsed = urlparse(url)
        if parsed.path.lower().endswith(".pdf") and parsed.netloc in PDF_HOSTS \
                and url not in links:
            links.append(url)
    return links


def pdf_title(url: str) -> str:
    """'.../Table_1A-A_-_Building_Permit_Fees_2026.pdf' -> 'Table 1A-A - Building Permit Fees 2026'"""
    name = unquote(Path(urlparse(url).path).name)
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    # sf.gov appends a random 7-character suffix to some file names (_l3tFiXs)
    name = re.sub(r"_(?=[A-Za-z]*\d)[A-Za-z0-9]{7}$", "", name)
    return _clean(name.replace("_", " "))


def parse_pdf(content: bytes) -> list[dict]:
    """One section per PDF page; each text line is a paragraph."""
    reader = PdfReader(io.BytesIO(content))
    multi_page = len(reader.pages) > 1
    sections = []
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            continue  # unreadable page; keep the rest of the document
        lines = [_clean(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        if lines:
            sections.append({"heading": f"Page {number}" if multi_page else "",
                             "paragraphs": lines})
    return sections


def _get_allowed(session: requests.Session, robots: Robots, url: str,
                 headers: dict) -> Optional[requests.Response]:
    """
    GET that checks robots.txt on every redirect hop. None if disallowed.
    Waits REQUEST_DELAY_SECONDS before each request actually sent.
    """
    for _ in range(MAX_REDIRECTS + 1):
        if not robots.allows(url):
            return None
        time.sleep(REQUEST_DELAY_SECONDS)
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS,
                               allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, response.headers["Location"])
            continue
        return response
    raise RuntimeError(f"more than {MAX_REDIRECTS} redirects")


def _word_count(page: dict) -> int:
    return sum(len(p.split()) for s in page["sections"] for p in s["paragraphs"])


def _content_hash(page: dict) -> str:
    payload = json.dumps([page["title"], page["sections"]], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def url_to_filename(url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", urlparse(url).path.lower()).strip("-")[:120]
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}.json"


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def _load_previous_manifest(raw_dir: Path) -> dict:
    manifest_file = raw_dir / "manifest.json"
    if manifest_file.exists():
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    return {}


def _conditional_headers(prev: Optional[dict], prev_file: Optional[Path]) -> dict:
    headers = {}
    if prev and prev_file and prev_file.exists():
        if prev.get("etag"):
            headers["If-None-Match"] = prev["etag"]
        if prev.get("last_modified"):
            headers["If-Modified-Since"] = prev["last_modified"]
    return headers


def crawl(urls: dict[str, str], out_dir: Path, previous_dir: Optional[Path]) -> dict:
    """Fetch every URL (and the PDFs they link) into out_dir/pages. Returns the manifest."""
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)
    robots = Robots(session)
    previous_manifest = _load_previous_manifest(previous_dir) if previous_dir else {}
    previous = previous_manifest.get("pages", {})

    manifest_pages: dict[str, dict] = {}
    stats = {"saved": 0, "unchanged": 0, "skipped_empty": 0,
             "skipped_robots": 0, "failed": 0}
    failures: list[str] = []
    pdf_links: dict[str, str] = {}  # pdf url -> category of the first page linking it

    for i, (url, category) in enumerate(urls.items(), 1):
        prefix = f"[{i}/{len(urls)}]"
        if not robots.allows(url):
            stats["skipped_robots"] += 1
            print(f"{prefix} • robots.txt disallows {url}")
            continue

        prev = previous.get(url)
        prev_file = (previous_dir / "pages" / prev["file"]) if prev and previous_dir else None
        conditional = _conditional_headers(prev, prev_file)

        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            response = session.get(url, headers=conditional, timeout=REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 304 and prev_file:
                shutil.copy2(prev_file, pages_dir / prev["file"])
                manifest_pages[url] = prev
                for pdf in prev.get("pdf_links", []):
                    pdf_links.setdefault(pdf, category)
                stats["unchanged"] += 1
                print(f"{prefix} = unchanged  {url}")
                continue

            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", "").lower():
                stats["skipped_empty"] += 1
                print(f"{prefix} • not HTML  {url}")
                continue

            parsed = parse_page(response.content)
            if _word_count(parsed) < MIN_SECTION_WORDS_PER_PAGE:
                stats["skipped_empty"] += 1
                print(f"{prefix} • too little text  {url}")
                continue

            page = {
                "url": url,
                "category": category,
                "title": parsed["title"],
                "sections": parsed["sections"],
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            page["content_hash"] = _content_hash(page)
            filename = url_to_filename(url)
            (pages_dir / filename).write_text(
                json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8")

            links = extract_pdf_links(response.content, url)
            for pdf in links:
                pdf_links.setdefault(pdf, category)

            changed = not prev or prev.get("content_hash") != page["content_hash"]
            manifest_pages[url] = {
                "file": filename,
                "category": category,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_hash": page["content_hash"],
                "pdf_links": links,
            }
            stats["saved" if changed else "unchanged"] += 1
            print(f"{prefix} {'✓ saved    ' if changed else '= unchanged'} {url} "
                  f"({_word_count(parsed)} words)")

        except Exception as e:
            stats["failed"] += 1
            failures.append(f"{url}: {e}")
            print(f"{prefix} ✗ {url}: {e}")

    manifest_pdfs, pdf_stats = crawl_pdfs(
        pdf_links, session, robots, pages_dir,
        previous_manifest.get("pdfs", {}), previous_dir, failures)

    per_category: dict[str, int] = {topic: 0 for topic in TOPICS}
    for entry in manifest_pages.values():
        per_category[entry["category"]] += 1

    manifest = {
        "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SITEMAP_URL,
        "urls_selected": len(urls),
        "stats": stats,
        "pages_per_category": per_category,
        "pdf_stats": pdf_stats,
        "failures": failures,
        "pages": manifest_pages,
        "pdfs": manifest_pdfs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def crawl_pdfs(pdf_links: dict[str, str], session: requests.Session, robots: Robots,
               pages_dir: Path, previous: dict, previous_dir: Optional[Path],
               failures: list[str]) -> tuple[dict, dict]:
    """Fetch linked PDFs into pages_dir. PDF failures are reported, not fatal."""
    manifest_pdfs: dict[str, dict] = {}
    stats = {"saved": 0, "unchanged": 0, "skipped_empty": 0,
             "skipped_robots": 0, "skipped_too_large": 0, "failed": 0}

    for i, (url, category) in enumerate(sorted(pdf_links.items()), 1):
        prefix = f"[PDF {i}/{len(pdf_links)}]"
        prev = previous.get(url)
        prev_file = (previous_dir / "pages" / prev["file"]) if prev and previous_dir else None

        try:
            response = _get_allowed(session, robots, url,
                                    _conditional_headers(prev, prev_file))
            if response is None:
                stats["skipped_robots"] += 1
                print(f"{prefix} • robots.txt disallows {url}")
                continue
            if response.status_code == 304 and prev_file:
                shutil.copy2(prev_file, pages_dir / prev["file"])
                manifest_pdfs[url] = prev
                stats["unchanged"] += 1
                print(f"{prefix} = unchanged  {url}")
                continue
            response.raise_for_status()

            content = response.content
            if len(content) > MAX_PDF_BYTES:
                stats["skipped_too_large"] += 1
                print(f"{prefix} • too large ({len(content):,} bytes)  {url}")
                continue
            if not content.startswith(b"%PDF"):
                stats["skipped_empty"] += 1
                print(f"{prefix} • not a PDF  {url}")
                continue

            page = {
                "url": url,
                "category": category,
                "title": pdf_title(url),
                "sections": parse_pdf(content),
                "doc_type": "pdf",
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if _word_count(page) < MIN_SECTION_WORDS_PER_PAGE:
                stats["skipped_empty"] += 1  # usually a scanned PDF without a text layer
                print(f"{prefix} • too little text  {url}")
                continue

            page["content_hash"] = _content_hash(page)
            filename = url_to_filename(url)
            (pages_dir / filename).write_text(
                json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest_pdfs[url] = {
                "file": filename,
                "category": category,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_hash": page["content_hash"],
            }
            stats["saved"] += 1
            print(f"{prefix} ✓ saved     {url} ({_word_count(page)} words)")

        except Exception as e:
            stats["failed"] += 1
            failures.append(f"{url}: {e}")
            print(f"{prefix} ✗ {url}: {e}")

    return manifest_pdfs, stats


def check_health(manifest: dict) -> list[str]:
    """Return a list of problems that should stop data/raw from being replaced."""
    problems = []
    attempted = manifest["urls_selected"] - manifest["stats"]["skipped_robots"]
    if attempted and manifest["stats"]["failed"] / attempted > MAX_FAILURE_RATE:
        problems.append(
            f"{manifest['stats']['failed']} of {attempted} fetches failed "
            f"(limit {MAX_FAILURE_RATE:.0%})")
    for topic, count in manifest["pages_per_category"].items():
        if count == 0:
            problems.append(f"topic '{topic}' has 0 pages")
    return problems


def swap_into_place(staging: Path, target: Path) -> None:
    """Replace target with staging, keeping one backup at <target>_prev."""
    backup = target.with_name(target.name + "_prev")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    staging.rename(target)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("📚 COLLECTING SF.GOV PAGES (sitemap-based)")
    print("=" * 70)

    session = requests.Session()
    session.headers.update(HEADERS)
    urls = discover_urls(session)
    per_topic: dict[str, int] = {}
    for category in urls.values():
        per_topic[category] = per_topic.get(category, 0) + 1
    print(f"Selected {len(urls)} pages from the sitemap: {per_topic}\n")

    staging = DATA_DIR / f"raw_staging_{datetime.now():%Y%m%d_%H%M%S}"
    manifest = crawl(urls, staging, previous_dir=RAW_DIR if RAW_DIR.exists() else None)

    print("\n" + "=" * 70)
    print(f"Stats: {manifest['stats']}")
    print(f"Pages per topic: {manifest['pages_per_category']}")
    print(f"PDFs: {manifest['pdf_stats']}")

    problems = check_health(manifest)
    if problems:
        print("\n❌ Crawl looks unhealthy — data/raw was NOT replaced:")
        for problem in problems:
            print(f"   • {problem}")
        print(f"   Partial results kept for inspection in {staging}")
        return 1

    swap_into_place(staging, RAW_DIR)
    print(f"\n✅ data/raw replaced (previous copy kept in {RAW_DIR.name}_prev)")
    print("Next: python process_docs.py && python build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
