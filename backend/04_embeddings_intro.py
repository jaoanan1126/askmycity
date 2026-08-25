"""
Day 4: Introduction to embeddings
Learning: How text becomes meaningful numbers
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model (runs locally, free!)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded\n")

def basic_embeddings():
    """See what embeddings look like"""
    
    print("="*70)
    print("DEMO 1: What Are Embeddings?")
    print("="*70 + "\n")
    
    sentences = [
        "I love cats",
        "I adore kittens",
        "Dogs are great pets"
    ]
    
    print("Creating embeddings for:")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. \"{s}\"")
    print()
    
    # Create embeddings
    embeddings = model.encode(sentences)
    
    print(f"✓ Created embeddings!")
    # shows dimennsions, 3 is number of texts, embeding size
    print(f"  Shape: {embeddings.shape}")
    print(f"  Each sentence is now a {embeddings.shape[1]}-dimensional vector\n")
    
    # Show first sentence's embedding (first 10 dimensions)
    print(f"First sentence embedding (first 10 numbers):")
    print(f"  {embeddings[0][:10]}")
    print(f"\nThese {embeddings.shape[1]} numbers capture the MEANING of the text!\n")
    
    input("Press Enter to continue...\n")

def similarity_demo():
    """Demonstrate semantic similarity"""
    
    print("="*70)
    print("DEMO 2: Semantic Similarity")
    print("="*70 + "\n")
    
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",      # Same meaning!
        "The dog played in the park",
        "Python is a programming language",
        "Machine learning uses neural networks"
    ]
    
    print("Sentences:")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print()
    
    # Create embeddings
    embeddings = model.encode(sentences)
    
    # Calculate similarities
    similarities = cosine_similarity(embeddings)
    
    print("Similarity Matrix (1.0 = identical, 0.0 = unrelated):")
    print("-" * 70)
    
    # Show similarities for sentence 1
    """
    ## Visual representation:
    ```
            Sent 0   Sent 1   Sent 2
    Sent 0  [  1.000,   0.856,   0.234 ]
    Sent 1  [  0.856,   1.000,   0.189 ]
    Sent 2  [  0.234,   0.189,   1.000 ]
    """
    print(f"\nSentence 1: \"{sentences[0]}\"")
    print("Similarities with other sentences:\n")
    print(similarities)
    for j in range(1, len(sentences)):
        score = similarities[0][j]
        emoji = "🔥" if score > 0.6 else "✓" if score > 0.3 else "○"
        print(f"  {emoji} {score:.3f} - \"{sentences[j]}\"")
    
    print("\n🔥 Notice: Sentence 2 has HIGH similarity (0.6+)")
    print("   Even though it uses DIFFERENT WORDS!")
    print("   \"cat/mat\" vs \"feline/rug\" - same meaning!\n")
    
    input("Press Enter to continue...\n")

def keyword_vs_semantic():
    """Compare keyword search vs semantic search"""
    
    print("="*70)
    print("DEMO 3: Keyword Search vs Semantic Search")
    print("="*70 + "\n")
    
    documents = [
        "How to apply for a building permit in San Francisco",
        "Parking regulations in downtown SF",
        "Zoning laws for residential construction",
        "Business license requirements",
        "Report a pothole or street damage",
        "Apply for a street tree permit",
        "Food truck permit application process"
    ]
    
    query = "I want to build an extension on my house"
    
    print(f"📄 Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    print(f"\n❓ Query: \"{query}\"\n")
    
    # ===== KEYWORD SEARCH =====
    print("="*70)
    print("METHOD 1: KEYWORD SEARCH (old way)")
    print("="*70 + "\n")
    
    query_words = set(query.lower().split())
    print(f"Query words: {query_words}\n")
    
    print("Results:")
    for i, doc in enumerate(documents, 1):
        doc_words = set(doc.lower().split())
        overlap = query_words.intersection(doc_words)
        score = len(overlap) / len(query_words) if query_words else 0
        
        emoji = "✓" if overlap else "✗"
        print(f"  {emoji} Doc {i}: score={score:.2f}, matched words: {overlap if overlap else 'NONE'}")
    
    print("\n❌ Problem: No exact word matches!")
    print("   But doc #1 (building permit) IS the most relevant!\n")
    
    input("Press Enter for semantic search...\n")
    
    # ===== SEMANTIC SEARCH =====
    print("="*70)
    print("METHOD 2: SEMANTIC SEARCH (embeddings!)")
    print("="*70 + "\n")
    
    # Create embeddings
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Sort by relevance
    ranked_indices = np.argsort(similarities)[::-1]
    
    print("Results (ranked by relevance):")
    for rank, idx in enumerate(ranked_indices, 1):
        score = similarities[idx]
        emoji = "🎯" if rank == 1 else "✓"
        print(f"  {emoji} #{rank}: score={score:.3f} - {documents[idx]}")
    
    print("\n✅ Success! Found building permit doc as most relevant!")
    print("   Even though query said 'build extension' not 'building permit'")
    print("   Embeddings understand: building extension = need permit!\n")
    
    input("Press Enter to continue...\n")

def embedding_arithmetic():
    """Fun with embedding math!"""
    
    print("="*70)
    print("DEMO 4: Embedding Arithmetic 🤯")
    print("="*70 + "\n")
    
    print("Famous example: king - man + woman = ?\n")
    
    words = ["king", "man", "woman", "queen", "prince", "princess", "boy", "girl"]
    
    # Create embeddings
    embeddings_dict = {}
    embeddings_list = []
    
    for word in words:
        emb = model.encode([word])[0]
        embeddings_dict[word] = emb
        embeddings_list.append(emb)
    
    # The math: king - man + woman
    result = embeddings_dict["king"] - embeddings_dict["man"] + embeddings_dict["woman"]
    
    # Find closest word
    similarities = cosine_similarity([result], embeddings_list)[0]
    
    print("Calculation: king - man + woman\n")
    print("Closest words:")
    
    # Sort by similarity
    word_scores = list(zip(words, similarities))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    for word, score in word_scores:
        emoji = "👑" if word == "queen" else "✓"
        print(f"  {emoji} {word:10s} similarity: {score:.3f}")
    
    print("\n🤯 The math captured the relationship!")
    print("   'king' is to 'man' as 'queen' is to 'woman'\n")
    
    input("Press Enter to continue...\n")

def practical_semantic_search():
    """Practical semantic search system"""
    
    print("="*70)
    print("DEMO 5: Interactive Semantic Search")
    print("="*70 + "\n")
    
    knowledge_base = [
        "To apply for a building permit, submit plans to the Planning Department with a $500 fee.",
        "Parking meters operate Monday-Saturday, 9 AM to 6 PM. Sunday is free.",
        "Business licenses must be renewed annually by January 31st.",
        "Zoning regulations vary by neighborhood. Check the zoning map online.",
        "Report potholes through the 311 app or by calling 311.",
        "Food truck permits require health inspection and $1000 annual fee.",
        "Tree planting permits are free but require approval from Public Works.",
        "The Board of Supervisors meets every Tuesday at 2 PM in City Hall.",
        "Voter registration deadline is 15 days before election day.",
        "Public records requests must be submitted in writing to the department."
    ]
    
    print("📚 Knowledge Base loaded!")
    print(f"   {len(knowledge_base)} documents\n")
    
    # Create embeddings
    print("Creating embeddings...")
    kb_embeddings = model.encode(knowledge_base)
    print("✓ Ready for search!\n")
    
    print("Try searching! Type 'quit' to exit.\n")
    
    while True:
        query = input("🔍 Search: ").strip()
        
        if not query or query.lower() == 'quit':
            break
        
        # Search
        query_emb = model.encode([query])
        similarities = cosine_similarity(query_emb, kb_embeddings)[0]
        
        # Get top 3
        top_3 = np.argsort(similarities)[::-1][:3]
        
        print(f"\n📊 Top 3 results for: \"{query}\"\n")
        
        for rank, idx in enumerate(top_3, 1):
            score = similarities[idx]
            emoji = "🎯" if rank == 1 else "✓"
            print(f"{emoji} {rank}. [Similarity: {score:.3f}]")
            print(f"   {knowledge_base[idx]}\n")
        
    print("Thanks for searching! 👋\n")

if __name__ == "__main__":
    print("🎨 Day 4: Understanding Embeddings\n")
    
    basic_embeddings()
    similarity_demo()
    keyword_vs_semantic()
    embedding_arithmetic()
    practical_semantic_search()
    
    print("="*70)
    print("✅ Day 4 Complete!")
    print("="*70)
    print("\nYou now understand:")
    print("  ✓ What embeddings are (text → meaningful numbers)")
    print("  ✓ How semantic similarity works")
    print("  ✓ Why semantic search beats keyword search")
    print("  ✓ Embedding arithmetic captures relationships")
    print("  ✓ How to build practical search systems")
    print("\nTomorrow: Vector databases to scale this up!")