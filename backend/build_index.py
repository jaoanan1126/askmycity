"""
Step 3: Build the ChromaDB search index from data/processed/chunks.json

Builds into a temporary collection, checks it, then swaps it in:
    sf_government            <- new index (what the API reads)
    sf_government__previous  <- the index it replaced (one backup kept)

If anything fails before the swap, the live index is untouched.
Restart the API server afterwards so it picks up the new collection.

Usage:
    python build_index.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

# Must match api/rag_service.py
CHROMA_PATH = os.path.expanduser("~/Library/Application Support/askmycity/chroma_db")
COLLECTION = "sf_government"
BUILDING = f"{COLLECTION}__building"
PREVIOUS = f"{COLLECTION}__previous"

BATCH_SIZE = 100
SANITY_QUERY = "How do I apply for a building permit?"


def _exists(client: chromadb.ClientAPI, name: str) -> bool:
    return name in {c.name for c in client.list_collections()}


def build_index() -> int:
    if not CHUNKS_FILE.exists():
        print(f"❌ {CHUNKS_FILE} not found. Run process_docs.py first.")
        return 1
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if not chunks:
        print("❌ chunks.json is empty — refusing to replace the index.")
        return 1

    print(f"📥 Indexing {len(chunks)} chunks into {CHROMA_PATH}")
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = ONNXMiniLM_L6_V2()

    if _exists(client, BUILDING):  # leftover from an interrupted run
        client.delete_collection(BUILDING)
    collection = client.create_collection(
        name=BUILDING,
        embedding_function=embed_fn,
        # rag_service converts distance to similarity with 1 - distance,
        # which is only meaningful for cosine distance.
        configuration={"hnsw": {"space": "cosine"}},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"   • {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    # Checks before going live
    if collection.count() != len(chunks):
        print(f"❌ Expected {len(chunks)} chunks, index has {collection.count()}. "
              f"Live index untouched.")
        return 1
    result = collection.query(query_texts=[SANITY_QUERY], n_results=1)
    if not result["ids"][0]:
        print("❌ Sanity query returned nothing. Live index untouched.")
        return 1
    top = result["metadatas"][0][0]
    print(f"   Sanity query → {top.get('title')} › {top.get('section')} ({top.get('url')})")

    # Swap: live -> previous, building -> live
    if _exists(client, PREVIOUS):
        client.delete_collection(PREVIOUS)
    if _exists(client, COLLECTION):
        client.get_collection(COLLECTION).modify(name=PREVIOUS)
    collection.modify(name=COLLECTION)

    print(f"\n✅ '{COLLECTION}' now has {len(chunks)} chunks "
          f"(old index kept as '{PREVIOUS}').")
    print("   Restart the API server to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(build_index())
