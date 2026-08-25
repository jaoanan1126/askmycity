"""
Step 1: Collect SF Government Documents
Learn: Web crawling HTML pages from sf.gov
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from pathlib import Path
from typing import Union
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# SF Government URLs to scrape (crawl seeds)
SF_URLS = {
    "building_permits": "https://www.sf.gov/topics--building-permits",
    "otc_interior_remodel": "https://www.sf.gov/step-by-step--get-otc-permit-your-interior-residential-remodel",
    "business_license": "https://www.sf.gov/step-by-step--step-step-guide-starting-business-san-francisco",
    "parking": "https://www.sf.gov/topics--parking",
    "planning": "https://sf.gov/departments/planning",
    "public_works": "https://sf.gov/departments/public-works",
}

MAX_DEPTH = 3
MAX_PAGES_PER_SEED = 40
REQUEST_DELAY_SECONDS = 0.75
MIN_BODY_CHARS = 2500
ALLOWED_HOSTS = {"sf.gov", "www.sf.gov"}
SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".mp4", ".mp3", ".css", ".js",
}
SKIP_PATH_PREFIXES = (
    "/search",
    "/user/login",
    "/user/register",
    "/jobs",
    "/contact",
)
SKIP_EXACT_PATHS = {
    "/",
    "/services",
    "/departments",
}
# Hub/listing pages: follow links, but do not save as documents
LISTING_PATH_PREFIXES = (
    "/topics--",
    "/topics-",
    "/topic-",
)
LANG_PATH_PREFIXES = (
    "/es-",
    "/fil-",
    "/zh-hant-",
    "/zh-hans-",
    "/vi-vn-",
)
USER_AGENT = "AskMyCityBot/1.0 (educational RAG project)"
HEADERS = {"User-Agent": USER_AGENT}

def setup_directories():
    """Create data directories"""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("✓ Created data directories\n")

def _normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower() or 'https'}://{host}{path}"


def _same_site(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_HOSTS


def _should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    if not _same_site(url):
        return True
    path = (parsed.path or "").lower()
    if path in SKIP_EXACT_PATHS:
        return True
    if any(path.startswith(prefix) for prefix in SKIP_PATH_PREFIXES):
        return True
    suffix = Path(path).suffix
    if suffix in SKIP_EXTENSIONS:
        return True
    if any(path.startswith(prefix) for prefix in LANG_PATH_PREFIXES):
        return True
    return False


def _is_listing_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    if path in ("/", "/services", "/departments"):
        return True
    return any(path.startswith(prefix) for prefix in LISTING_PATH_PREFIXES)


def _url_to_filename(seed_name: str, url: str) -> str:
    parsed = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "-", parsed.path.lower()).strip("-") or "index"
    slug = slug[:80]
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{seed_name}__{slug}__{digest}.txt"


def _main_content(html: Union[bytes, str]) -> BeautifulSoup:
    """Use article body only so global menu links are not crawled or counted as body text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(id="main-content")
        or soup.find(id="content")
        or soup.find(class_="main-content")
    )
    return main if main is not None else soup


def html_to_text(html: Union[bytes, str]) -> str:
    node = _main_content(html)
    text = node.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_links(html: Union[bytes, str], base_url: str) -> list[str]:
    node = _main_content(html)
    links = []
    seen_hrefs: set[str] = set()
    for a in node.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        normalized = _normalize_url(absolute)
        if normalized in seen_hrefs or _should_skip_url(normalized):
            continue
        seen_hrefs.add(normalized)
        links.append(normalized)
    return links


def _load_robots(seed_url: str) -> RobotFileParser:
    parsed = urlparse(seed_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp


def scrape_webpage(url, name):
    """Scrape text from a single webpage (no link following)."""
    try:
        print(f"📥 Scraping {name}...")
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        text = html_to_text(response.content)
        filepath = f"data/raw/{name}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"   ✓ Saved to {filepath} ({len(text)} chars)\n")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False


def crawl_from_seed(seed_url: str, seed_name: str) -> int:
    """BFS crawl same-site HTML pages up to MAX_DEPTH from a parent URL."""
    print(f"🕷 Crawling {seed_name} from {seed_url} (depth ≤ {MAX_DEPTH}, cap {MAX_PAGES_PER_SEED})")
    robots = _load_robots(seed_url)
    queue = deque([( _normalize_url(seed_url), 0 )])
    seen: set[str] = set()
    saved = 0

    while queue and saved < MAX_PAGES_PER_SEED:
        url, depth = queue.popleft()
        if url in seen or depth > MAX_DEPTH or _should_skip_url(url):
            continue
        seen.add(url)

        if not robots.can_fetch(USER_AGENT, url):
            print(f"   • skipped (robots.txt): {url}")
            continue

        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type:
                continue

            text = html_to_text(response.content)
            too_thin = len(text) < MIN_BODY_CHARS
            listing = _is_listing_url(url)

            if too_thin or listing:
                reason = "listing" if listing else f"thin ({len(text)} chars)"
                print(f"   • skipped ({reason}): {url}")
            else:
                filename = _url_to_filename(seed_name, url)
                filepath = Path("data/raw") / filename
                header = f"SOURCE_URL: {url}\nSEED: {seed_name}\nDEPTH: {depth}\n\n"
                filepath.write_text(header + text, encoding="utf-8")
                saved += 1
                print(f"   ✓ [{depth}] {filename} ({len(text)} chars)")

            if depth < MAX_DEPTH:
                for link in _extract_links(response.content, url):
                    if link not in seen:
                        queue.append((link, depth + 1))
        except Exception as e:
            print(f"   ✗ {url}: {e}")

    print(f"   → {saved} pages saved for {seed_name}\n")
    return saved


def create_sample_documents():
    """
    Create sample SF government documents
    (Use this if web scraping fails)
    """
    print("📝 Creating sample documents...\n")
    
    sample_docs = {
        "building_permits": """
Building Permits in San Francisco

To apply for a building permit in San Francisco, you must submit the following:

1. Completed application form
2. Architectural plans drawn to scale
3. Site plan showing property boundaries
4. Structural calculations (if applicable)
5. Energy compliance documentation
6. Application fee: $500 for residential, $1000 for commercial

Submit documents to:
San Francisco Planning Department
49 South Van Ness Avenue, Suite 1400
San Francisco, CA 94103

Hours: Monday-Friday, 8:00 AM - 5:00 PM

Processing time: 4-6 weeks for standard permits

For questions, call 311 or visit sf.gov/planning

Special Requirements:
- Historic buildings require additional review
- Projects over 5,000 sq ft need environmental review
- Seismic retrofitting may be required for older buildings
""",
        
        "business_license": """
Starting a Business in San Francisco

Steps to start a business in San Francisco:

1. Register your business name
   - Check name availability at sf.gov
   - File with Secretary of State

2. Obtain Business Registration Certificate
   - Apply online or in person
   - Cost: $91-250 depending on business type
   - Renew annually

3. Obtain necessary permits:
   - Health permit (food businesses)
   - Building permit (physical changes)
   - Sign permit (exterior signage)

4. Register for taxes:
   - Federal EIN from IRS
   - State seller's permit (if selling goods)
   - City business tax registration

Contact:
Office of Small Business
1 Dr Carlton B Goodlett Place, Room 140
Phone: 415-554-6134
Email: smallbusiness@sfgov.org

Resources:
- Free business counseling
- Permit assistance
- Startup workshops

Special Programs:
- Microloan program (up to $50,000)
- Storefront improvement grants
- Technical assistance for underserved communities
""",
        
        "parking": """
Parking Regulations in San Francisco

Meter Parking:
- Operating hours: Monday-Saturday, 9 AM - 6 PM
- Sunday: FREE (except special event areas)
- Rates: $2-8 per hour depending on neighborhood
- Payment: Coin, credit card, or PayByPhone app
- Maximum stay: 2-4 hours (varies by location)

Residential Parking Permits:
- Cost: $147 per year
- Available to residents of permit areas
- Apply online at sf.gov/parking
- Proof of residency required
- One permit per vehicle

Parking Violations:
- Expired meter: $76
- Street cleaning: $76  
- Blocking driveway: $118
- Fire hydrant: $118
- Disabled zone: $976

Appeal Process:
- Online at sfmta.com
- Within 21 days of citation
- Photo evidence accepted

Parking Lots & Garages:
- 5th & Mission Garage: $3/hour
- Embarcadero Center: $4/hour
- Monthly parking: $200-400

Contact:
SFMTA Customer Service
One South Van Ness Avenue, 7th Floor
Phone: 311
""",
        
        "zoning": """
Zoning Regulations in San Francisco

San Francisco Zoning Districts:

Residential Districts:
- RH-1: Single-family homes
- RH-2: Two-family homes
- RH-3: Three-family homes
- RM: Mixed residential (up to 6 units)

Commercial Districts:
- NC: Neighborhood commercial
- C-3: Downtown commercial
- PDR: Production, distribution, repair

Height Limits:
- Residential areas: 40 feet (varies by district)
- Commercial corridors: 40-65 feet
- Downtown: Up to 550 feet (special approval)

Setback Requirements:
- Front yard: 5-15 feet (varies)
- Rear yard: 25% of lot depth
- Side yard: 4 feet minimum

Parking Requirements:
- Residential: 1 space per unit (some exceptions)
- Commercial: Varies by use
- Transit-oriented areas: Reduced requirements

Variances:
- Apply to Board of Appeals
- Public hearing required
- Fee: $500-2000
- 30-60 day process

Contact Planning Department:
49 South Van Ness Ave, Suite 1400
Phone: 415-558-6378
Email: planning@sfgov.org

Check zoning for your property:
sf.gov/check-zoning-my-property
""",
        
        "public_services": """
San Francisco Public Services

311 Customer Service Center:
- Call 311 (or 415-701-2311)
- Available 24/7
- Report issues:
  * Potholes
  * Graffiti
  * Abandoned vehicles
  * Streetlight outages
  * Tree maintenance

SF311 Mobile App:
- Report issues with photos
- Track request status
- Available: iOS and Android

Trash & Recycling:
- Collection: Weekly
- Recology: 415-330-1300
- Bulky item pickup: Schedule online
- Hazardous waste: Drop-off at Tunnel Ave

Public Libraries:
- Main Library: 100 Larkin Street
- 27 branch locations
- Free library card
- Hours: vary by branch
- sf.gov/libraries

Recreation & Parks:
- 220+ parks
- Permits for events
- Sports field reservations
- Recreation centers: sf.gov/rec-and-park

Public Transportation:
- Muni buses, trains, cable cars
- BART: Regional transit
- Ferry service
- Caltrain: Peninsula connection

Contact:
City Hall
1 Dr Carlton B Goodlett Place
San Francisco, CA 94102
Phone: 311
"""
    }
    
    for name, content in sample_docs.items():
        filepath = f"data/raw/{name}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ Created {filepath}")
    
    print(f"\n✓ Created {len(sample_docs)} sample documents\n")

def collect_all_documents():
    """Main collection function"""
    print("="*70)
    print("📚 COLLECTING SF GOVERNMENT DOCUMENTS")
    print("="*70 + "\n")
    
    setup_directories()
    
    print("Crawling live sf.gov pages (same-site, up to 3 hops from each parent)...\n")
    success_count = 0

    for name, url in SF_URLS.items():
        success_count += crawl_from_seed(url, name)
    
    # Option 2: Use sample documents if scraping failed
    if success_count == 0:
        print("⚠️ Web scraping didn't work (this is normal!)")
        print("Using pre-written sample documents instead...\n")
        create_sample_documents()
    
    # Count collected documents
    raw_files = list(Path("data/raw").glob("*.txt"))
    print("="*70)
    print(f"✅ COLLECTION COMPLETE: {len(raw_files)} documents")
    print("="*70)
    
    for f in raw_files:
        size = f.stat().st_size
        print(f"   • {f.name} ({size:,} bytes)")
    
    print()

if __name__ == "__main__":
    collect_all_documents()