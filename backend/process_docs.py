"""
Step 2: Process and chunk documents
Learn: Text cleaning, chunking strategies, metadata extraction
"""
import os
from pathlib import Path
import json
import re

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    
    # Fix common issues
    text = text.replace('\n', ' ')
    text = text.strip()
    
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    
    Why overlap? So context isn't lost at chunk boundaries!
    
    Example:
    Chunk 1: "...apply for a permit. You need..."
    Chunk 2: "You need to submit..." (overlaps!)
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Skip tiny chunks
            chunks.append(chunk.strip())
    
    return chunks

def extract_metadata(filepath, chunk_num, total_chunks):
    """Extract metadata from filename and content"""
    filename = Path(filepath).stem
    seed_name = filename.split("__")[0]

    # Infer category from seed / filename
    category_map = {
        "building_permits": "permits",
        "otc_interior_remodel": "permits",
        "business_license": "business",
        "parking": "transportation",
        "planning": "planning",
        "public_works": "services",
        "zoning": "planning",
        "public_services": "services"
    }

    return {
        "source": filename,
        "category": category_map.get(seed_name, category_map.get(filename, "general")),
        "chunk": chunk_num,
        "total_chunks": total_chunks,
        "doc_type": "government_doc"
    }

def process_all_documents():
    """Process all documents into chunks"""
    print("="*70)
    print("⚙️ PROCESSING DOCUMENTS")
    print("="*70 + "\n")
    
    raw_dir = Path("data/raw")
    processed_file = Path("data/processed/chunks.json")
    
    all_chunks = []
    
    # Process each document
    for filepath in raw_dir.glob("*.txt"):
        print(f"📄 Processing {filepath.name}...")
        
        # Read document
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean text
        cleaned = clean_text(text)
        
        # Split into chunks
        chunks = chunk_text(cleaned, chunk_size=500, overlap=50)
        
        print(f"   • Original: {len(text):,} chars")
        print(f"   • Cleaned: {len(cleaned):,} chars")
        print(f"   • Chunks: {len(chunks)}")
        
        # Create chunk objects with metadata
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "id": f"{filepath.stem}_chunk_{i}",
                "text": chunk,
                "metadata": extract_metadata(filepath, i, len(chunks))
            }
            all_chunks.append(chunk_obj)
        
        print(f"   ✓ Created {len(chunks)} chunks\n")
    
    # Save processed chunks
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print("="*70)
    print(f"✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Saved to: {processed_file}")
    print(f"   Average chunk size: {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars")
    print()
    
    # Show sample
    print("📋 Sample chunk:")
    print("-" * 70)
    sample = all_chunks[0]
    print(f"ID: {sample['id']}")
    print(f"Text: {sample['text'][:200]}...")
    print(f"Metadata: {json.dumps(sample['metadata'], indent=2)}")
    print()

if __name__ == "__main__":
    process_all_documents()