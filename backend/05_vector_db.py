"""
Day 5: Vector databases with ChromaDB
Learning: How to store and query embeddings at scale
"""
import chromadb
from chromadb.utils import embedding_functions
import json
import time

def demo_1_basic_chromadb():
    """Introduction to ChromaDB"""
    
    print("="*70)
    print("DEMO 1: Basic ChromaDB Operations")
    print("="*70 + "\n")
    
    # Create client (in-memory for demo)
    client = chromadb.Client()
    
    # Create collection with auto-embedding
    collection = client.create_collection(
        name="demo_collection",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    print("✓ Created collection\n")
    
    # Add documents
    documents = [
        "Claude is an AI assistant created by Anthropic",
        "Python is a popular programming language for AI",
        "Vector databases store embeddings for semantic search",
        "San Francisco is known for technology companies",
        "Machine learning models learn from data"
    ]
    
    print("Adding documents...")
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    print(f"✓ Added {len(documents)} documents\n")
    
    # Query
    queries = [
        "Tell me about AI assistants",
        "What programming language should I learn?",
        "What is San Francisco famous for?"
    ]
    
    for query in queries:
        print(f"Query: \"{query}\"")
        
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        print("Top 2 results:")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"  {i}. {doc}")
        print()
    
    input("Press Enter to continue...\n")

def demo_2_persistent_storage():
    """ChromaDB with persistent storage"""
    
    print("="*70)
    print("DEMO 2: Persistent ChromaDB")
    print("="*70 + "\n")
    
    # Create persistent client
    client = chromadb.PersistentClient(path="./demo_chroma_db")
    
    # Try to get existing collection
    try:
        collection = client.get_collection(
            name="government_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        print(f"✓ Loaded existing collection ({collection.count()} documents)\n")
        
        # Show it persisted!
        if collection.count() > 0:
            print("🎉 Data persisted from last run!")
            print("   Try running this script again - data will still be here!\n")
    
    except:
        # Create new collection
        collection = client.create_collection(
            name="government_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        print("✓ Created new collection\n")
    
    # Add documents if collection is empty
    if collection.count() == 0:
        gov_docs = [
            "To apply for a building permit, submit architectural plans and pay $500 fee at Planning Department.",
            "Parking meters operate Monday-Saturday 9 AM-6 PM. Rates vary by neighborhood ($2-8/hour).",
            "Business licenses must be renewed annually. Apply online or at City Hall.",
            "Zoning regulations differ by district. Check sf.gov/zoning for your address.",
            "Report potholes via 311 app or call 311. Include location and photos.",
            "Food truck permits require health inspection, business license, and mobile food facility permit.",
            "Street tree planting permits are free. Contact Public Works for approved species list.",
            "Board of Supervisors meets Tuesdays 2 PM at City Hall. Public comment allowed."
        ]
        
        print("Adding government documents...")
        collection.add(
            documents=gov_docs,
            ids=[f"gov_doc_{i}" for i in range(len(gov_docs))],
            metadatas=[
                {"type": "permit", "category": "building"},
                {"type": "regulation", "category": "parking"},
                {"type": "license", "category": "business"},
                {"type": "regulation", "category": "zoning"},
                {"type": "service", "category": "infrastructure"},
                {"type": "permit", "category": "business"},
                {"type": "permit", "category": "environment"},
                {"type": "meeting", "category": "civic"}
            ]
        )
        print(f"✓ Added {len(gov_docs)} documents\n")
    
    # Interactive search
    print("Try searching! (type 'quit' to exit)\n")
    
    while True:
        query = input("🔍 Query: ").strip()
        
        if not query or query.lower() == 'quit':
            break
        
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        print(f"\nTop 3 results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            print(f"\n{i}. {doc}")
            print(f"   📊 Metadata: {metadata}")
        
        print()
    
    print(f"\n✓ Data persisted to ./demo_chroma_db/")
    print("💡 Run this script again - your data will still be there!\n")
    
    input("Press Enter to continue...\n")

def demo_3_metadata_filtering():
    """Using metadata filters"""
    
    print("="*70)
    print("DEMO 3: Metadata Filtering")
    print("="*70 + "\n")
    
    client = chromadb.Client()
    
    collection = client.create_collection(
        name="filtered_docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    # Add documents with different categories
    documents = [
        "Building permits require architectural plans and fee payment",
        "Parking regulations vary by zone and time of day",
        "Starting a restaurant requires health permits and inspections",
        "Zoning laws affect building height and density",
        "Food truck vendors need mobile food facility permits",
        "Business tax rates depend on annual revenue",
        "Tree removal permits needed for trees over 6 inches diameter",
        "Noise ordinances restrict construction hours to 7 AM - 8 PM"
    ]
    
    metadata = [
        {"category": "permits", "type": "building", "cost": "high"},
        {"category": "transportation", "type": "parking", "cost": "none"},
        {"category": "permits", "type": "business", "cost": "medium"},
        {"category": "zoning", "type": "building", "cost": "none"},
        {"category": "permits", "type": "business", "cost": "high"},
        {"category": "business", "type": "tax", "cost": "variable"},
        {"category": "permits", "type": "environment", "cost": "low"},
        {"category": "regulation", "type": "building", "cost": "none"}
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=metadata
    )
    
    print("✓ Added documents with metadata\n")
    
    # Test 1: No filter
    query = "business requirements"
    
    print(f"Query: \"{query}\"")
    print("Filter: NONE\n")
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print("Results (no filter):")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"{i}. {doc[:50]}...")
        print(f"   {meta}\n")
    
    input("Press Enter to see filtered results...\n")
    
    # Test 2: With filter
    print(f"Query: \"{query}\"")
    print("Filter: category='permits'\n")
    
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"category": "permits"}
    )
    
    print("Results (permits only):")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"{i}. {doc[:50]}...")
        print(f"   {meta}\n")
    
    print("✨ Notice: Only documents with category='permits' returned!\n")
    
    input("Press Enter to continue...\n")

def demo_4_performance_at_scale():
    """Show ChromaDB performance at scale"""
    
    print("="*70)
    print("DEMO 4: Performance at Scale")
    print("="*70 + "\n")
    
    client = chromadb.Client()
    collection = client.create_collection(
        name="large_collection",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    # Generate synthetic documents
    n_docs = 1000
    print(f"📝 Generating {n_docs} synthetic documents...")
    
    doc_types = ["permit", "regulation", "service", "tax", "meeting"]
    documents = []
    metadatas = []
    
    for i in range(n_docs):
        doc_type = doc_types[i % len(doc_types)]
        documents.append(
            f"Document {i} about {doc_type} requirements, procedures, fees, and application process."
        )
        metadatas.append({
            "type": doc_type,
            "doc_id": i,
            "category": "government"
        })
    
    print("✓ Generated documents\n")
    
    # Add documents in batches
    print("📥 Adding to ChromaDB in batches...")
    start = time.time()
    
    batch_size = 100
    for i in range(0, n_docs, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_docs))]
        batch_meta = metadatas[i:i+len(batch_docs)]
        
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_meta
        )
        
        print(f"  Added {min(i+batch_size, n_docs)}/{n_docs}", end="\r")
    
    add_time = time.time() - start
    print(f"\n✓ Added {n_docs} documents in {add_time:.2f}s")
    print(f"  ({n_docs/add_time:.0f} docs/second)\n")
    
    # Query performance test
    query = "tell me about permit requirements"
    
    print(f"🔍 Searching {n_docs} documents...")
    print(f"   Query: \"{query}\"\n")
    
    # First search (includes embedding creation)
    start = time.time()
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    search_time_1 = time.time() - start
    
    print(f"✓ First search: {search_time_1*1000:.2f}ms")
    
    # Second search (embedding cached)
    start = time.time()
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    search_time_2 = time.time() - start
    
    print(f"✓ Second search: {search_time_2*1000:.2f}ms (faster!)")
    print(f"\n⚡ Searched {n_docs} documents in < {max(search_time_1, search_time_2)*1000:.0f}ms\n")
    
    print("Top 3 results:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0][:3], results['metadatas'][0][:3]), 1):
        print(f"  {i}. {doc[:60]}...")
        print(f"     Type: {meta['type']}, ID: {meta['doc_id']}\n")
    
    input("Press Enter to continue...\n")

def demo_5_update_and_delete():
    """Updating and deleting documents"""
    
    print("="*70)
    print("DEMO 5: Update & Delete Operations")
    print("="*70 + "\n")
    
    client = chromadb.Client()
    collection = client.create_collection(
        name="crud_demo",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    # Add initial documents
    documents = [
        "Old information about permits",
        "Outdated parking rates",
        "This will be deleted"
    ]
    
    collection.add(
        documents=documents,
        ids=["doc_1", "doc_2", "doc_3"]
    )
    
    print("✓ Added 3 documents\n")
    print("Initial documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()
    
    input("Press Enter to update document 1...\n")
    
    # UPDATE
    print("🔄 Updating doc_1...")
    collection.update(
        ids=["doc_1"],
        documents=["NEW: Updated permit information for 2025"]
    )
    
    results = collection.get(ids=["doc_1"])
    print(f"✓ Updated! New content: {results['documents'][0]}\n")
    
    input("Press Enter to delete document 3...\n")
    
    # DELETE
    print("🗑️ Deleting doc_3...")
    collection.delete(ids=["doc_3"])
    
    print(f"✓ Deleted! Collection now has {collection.count()} documents\n")
    
    # Show remaining
    all_docs = collection.get()
    print("Remaining documents:")
    for i, doc in enumerate(all_docs['documents'], 1):
        print(f"  {i}. {doc}")
    print()
    
    input("Press Enter to continue...\n")

def demo_6_production_patterns():
    """Production-ready patterns"""
    
    print("="*70)
    print("DEMO 6: Production Patterns")
    print("="*70 + "\n")
    
    client = chromadb.PersistentClient(path="./production_db")
    
    # Pattern 1: Safe collection creation
    def get_or_create_collection(client, name):
        """Get existing collection or create new one"""
        try:
            collection = client.get_collection(
                name=name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            print(f"✓ Loaded existing collection '{name}'")
            return collection
        except:
            collection = client.create_collection(
                name=name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            print(f"✓ Created new collection '{name}'")
            return collection
    
    collection = get_or_create_collection(client, "production_docs")
    print()
    
    # Pattern 2: Batch processing
    def add_documents_safely(collection, documents, batch_size=100):
        """Add documents in batches with error handling"""
        total = len(documents)
        
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            batch_ids = [f"doc_{j}" for j in range(i, i+len(batch))]
            
            try:
                collection.add(
                    documents=batch,
                    ids=batch_ids
                )
                print(f"✓ Added batch {i//batch_size + 1} ({len(batch)} docs)")
            except Exception as e:
                print(f"✗ Error in batch {i//batch_size + 1}: {e}")
    
    if collection.count() == 0:
        print("Adding sample documents...")
        sample_docs = [f"Sample document {i} about government services" for i in range(50)]
        add_documents_safely(collection, sample_docs)
        print()
    
    # Pattern 3: Search with error handling
    def search_safely(collection, query, n_results=3):
        """Search with error handling and validation"""
        if not query or not query.strip():
            return {"error": "Empty query"}
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count())
            )
            return {
                "success": True,
                "results": results['documents'][0],
                "count": len(results['documents'][0])
            }
        except Exception as e:
            return {"error": str(e)}
    
    print("\n💡 Production Patterns Demonstrated:")
    print("  ✓ Safe collection creation (get or create)")
    print("  ✓ Batch processing with error handling")
    print("  ✓ Search validation and error handling")
    print("  ✓ Persistent storage")
    print()
    
    # Test search
    result = search_safely(collection, "government services", n_results=3)
    
    if result.get("success"):
        print(f"✓ Search returned {result['count']} results\n")
    
    print("🎯 These patterns are ready for production use!\n")

if __name__ == "__main__":
    print("🗄️ Day 5: Vector Databases with ChromaDB\n")
    
    demo_1_basic_chromadb()
    demo_2_persistent_storage()
    demo_3_metadata_filtering()
    demo_4_performance_at_scale()
    demo_5_update_and_delete()
    demo_6_production_patterns()
    
    print("="*70)
    print("✅ Day 5 Complete!")
    print("="*70)
    print("\nYou now understand:")
    print("  ✓ ChromaDB basics (create, add, query)")
    print("  ✓ Persistent storage (data survives restarts)")
    print("  ✓ Metadata filtering (structured + semantic search)")
    print("  ✓ Performance at scale (1000s of docs)")
    print("  ✓ CRUD operations (update, delete)")
    print("  ✓ Production patterns (error handling, batching)")
    print("\n🎉 Ready to build RAG system this weekend!")