"""
Step 2: Turn crawled pages into search chunks

Reads data/raw/pages/*.json (written by collect_docs.py) and writes
data/processed/chunks.json.

Chunking strategy
-----------------
- Split along the page's own headings, never across sections.
- Pack whole paragraphs into chunks of ~TARGET_WORDS words; split an
  over-long paragraph at sentence boundaries.
- Start every chunk with "<page title> › <section heading>" so a chunk
  still makes sense on its own (and the heading words help search).
- Keep chunks under the embedding model's limit: all-MiniLM-L6-v2 only
  reads the first 256 tokens, so anything longer is invisible to search.
- Text is NOT stripped of symbols: "$76", "sf.gov/parking" and emails
  stay intact.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "data" / "raw" / "pages"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

TARGET_WORDS = 150      # aim for this many words per chunk body
MAX_WORDS = 180         # hard cap (~230 tokens + header stays under 256)
MIN_SECTION_WORDS = 40  # smaller sections are merged into the next one
OVERLAP_WORDS = 25      # carried into the next chunk of the SAME section

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> int:
    return len(text.split())


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Split a paragraph over MAX_WORDS at sentence (then word) boundaries."""
    if _words(paragraph) <= MAX_WORDS:
        return [paragraph]
    pieces, current = [], []
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if current and _words(" ".join(current + [sentence])) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    # A single sentence can still exceed the cap: hard-split by words.
    result = []
    for piece in pieces:
        words = piece.split()
        result.extend(" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS))
    return result


def _tail(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[-n_words:]) if len(words) > n_words else ""


def chunk_section(paragraphs: list[str]) -> list[str]:
    """Pack paragraphs into chunk bodies of about TARGET_WORDS words."""
    units = [u for p in paragraphs for u in _split_long_paragraph(p)]
    bodies: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n".join(current + [unit])
        if current and (_words(candidate) > MAX_WORDS or _words("\n".join(current)) >= TARGET_WORDS):
            body = "\n".join(current)
            bodies.append(body)
            overlap = _tail(body, OVERLAP_WORDS)
            current = [f"…{overlap}"] if overlap and _words(overlap) + _words(unit) <= MAX_WORDS else []
        current.append(unit)
    if current:
        bodies.append("\n".join(current))
    return bodies


def merge_small_sections(sections: list[dict]) -> list[dict]:
    """Merge sections under MIN_SECTION_WORDS into the following section."""
    merged: list[dict] = []
    pending: dict | None = None
    for section in sections:
        if pending:
            heading = "; ".join(h for h in (pending["heading"], section["heading"]) if h)
            section = {"heading": heading,
                       "paragraphs": pending["paragraphs"] + section["paragraphs"]}
            pending = None
        if _words(" ".join(section["paragraphs"])) < MIN_SECTION_WORDS:
            pending = section
        else:
            merged.append(section)
    if pending:  # small trailing section: append to the previous one if any
        if merged:
            merged[-1]["paragraphs"].extend(pending["paragraphs"])
        else:
            merged.append(pending)
    return merged


def chunk_page(page: dict, slug: str) -> list[dict]:
    """Turn one crawled page into chunk dicts ready for ChromaDB."""
    title = page["title"] or slug
    chunks = []
    for section in merge_small_sections(page["sections"]):
        header = f"{title} › {section['heading']}" if section["heading"] else title
        for body in chunk_section(section["paragraphs"]):
            chunks.append({"header": header, "section": section["heading"], "body": body})

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "id": f"{slug}__{i}",
            "text": f"{chunk['header']}\n\n{chunk['body']}",
            "metadata": {
                # ChromaDB metadata values must be str/int/float/bool
                "source": slug,
                "url": page["url"],
                "title": title,
                "section": chunk["section"],
                "category": page["category"],
                "chunk": i,
                "total_chunks": len(chunks),
                "fetched_at": page["fetched_at"],
                "doc_type": "government_doc",
            },
        })
    return result


def process_all_documents() -> int:
    print("=" * 70)
    print("⚙️  PROCESSING PAGES INTO CHUNKS")
    print("=" * 70 + "\n")

    page_files = sorted(PAGES_DIR.glob("*.json"))
    if not page_files:
        print(f"❌ No pages found in {PAGES_DIR}. Run collect_docs.py first.")
        return 1

    all_chunks: list[dict] = []
    for path in page_files:
        page = json.loads(path.read_text(encoding="utf-8"))
        slug = path.stem.split("__")[0]
        all_chunks.extend(chunk_page(page, slug))

    ids = [c["id"] for c in all_chunks]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        print(f"❌ {duplicates} duplicate chunk IDs — refusing to write {PROCESSED_FILE}")
        return 1

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    sizes = sorted(_words(c["text"]) for c in all_chunks)
    per_category: dict[str, int] = {}
    for c in all_chunks:
        per_category[c["metadata"]["category"]] = per_category.get(c["metadata"]["category"], 0) + 1

    print(f"✅ {len(page_files)} pages → {len(all_chunks)} chunks → {PROCESSED_FILE}")
    print(f"   Words per chunk: median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"   Chunks per topic: {per_category}")
    print("\nNext: python build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(process_all_documents())
