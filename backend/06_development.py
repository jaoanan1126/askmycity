"""
Step 3: Complete RAG System
Combines: ChromaDB + Claude + Everything we learned!
"""
print("loading anthropic...", flush=True)
import anthropic
print("loading chromadb core...", flush=True)
import chromadb
print("loading ONNX MiniLM (no PyTorch)...", flush=True)
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
print("loading stdlib...", flush=True)
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
print("imports done", flush=True)

# Keep Chroma off iCloud Desktop — PersistentClient writes many small files.
CHROMA_PATH = os.path.expanduser("~/Library/Application Support/askmycity/chroma_db")

class AskMyCityRAG:
    """
    Complete RAG system for SF government questions
    """
    
    def __init__(self):
        print("init: Claude client...", flush=True)
        self.claude = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        print(f"init: Chroma at {CHROMA_PATH}...", flush=True)
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        print("init: import onnxruntime...", flush=True)
        import onnxruntime  # noqa: F401
        print("init: import tokenizers...", flush=True)
        import tokenizers  # noqa: F401
        print("init: ONNXMiniLM_L6_V2()...", flush=True)
        embed_fn = ONNXMiniLM_L6_V2()

        print("init: get or create collection...", flush=True)
        try:
            self.collection = self.chroma_client.get_collection(
                name="sf_government",
                embedding_function=embed_fn
            )
            print(f"✓ Loaded existing collection ({self.collection.count()} documents)\n")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="sf_government",
                embedding_function=embed_fn
            )
            print("✓ Created new collection\n")
    
    def load_documents(self):
        """Load processed chunks into ChromaDB"""
        
        # Check if already loaded
        if self.collection.count() > 0:
            print(f"ℹ️  Documents already loaded ({self.collection.count()} chunks)")
            return
        
        print("📥 Loading documents into ChromaDB...\n")
        
        # Load processed chunks
        chunks_file = Path("data/processed/chunks.json")
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        # Prepare data for ChromaDB
        documents = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )
            
            print(f"   • Added batch {i//batch_size + 1} ({len(batch_docs)} chunks)")
        
        print(f"\n✅ Loaded {len(documents)} chunks into ChromaDB\n")
    
    def search(self, query, n_results=3, filter_category=None):
        """
        Search for relevant documents
        
        Args:
            query: User's question
            n_results: Number of results to return
            filter_category: Optional category filter
        """
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def ask(self, question):
        """
        Main RAG function: Search + Generate Answer
        """
        print(f"❓ Question: {question}\n")
        
        # Step 1: Search for relevant documents
        print("🔍 Searching knowledge base...")
        search_results = self.search(question, n_results=3)
        # print("results: ", search_results)
        
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        print(f"   ✓ Found {len(documents)} relevant chunks\n")
        
        # Step 2: Build context from search results
        context = ""
        sources = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            context += f"[Document {i}]\n{doc}\n\n"
            sources.append(f"{meta['source']} (chunk {meta['chunk']})")
        
        # Step 3: Generate answer with Claude
        print("🤖 Generating answer with Claude...\n")
        
        system_prompt = """You are a helpful assistant for San Francisco government services.
Answer questions based ONLY on the provided context documents.
Be specific, accurate, and cite which document you're using.
If the context doesn't contain the answer, say so."""
        
        user_message = f"""Context documents:

{context}

User question: {question}

Please provide a clear, helpful answer based on the context above. Include specific details like fees, locations, phone numbers when available."""
        
        # Stream the response
        print("💬 Answer:\n")
        full_response = ""
        
        with self.claude.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
        
        print("\n")
        
        # Show sources
        print("📚 Sources:")
        for source in sources:
            print(f"   • {source}")
        print()
        
        return {
            "question": question,
            "answer": full_response,
            "sources": sources,
            "context_docs": documents
        }
    
    def chat(self):
        """Interactive chat interface"""
        print("="*70)
        print("🏛️ ASKMYCITY - SF Government Assistant")
        print("="*70)
        print("\nAsk me anything about San Francisco government services!")
        print("Commands: 'quit' to exit, 'stats' for info\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'quit':
                    print("\nGoodbye! 👋\n")
                    break
                
                if question.lower() == 'stats':
                    print(f"\n📊 Statistics:")
                    print(f"   Documents in knowledge base: {self.collection.count()}")
                    print()
                    continue
                
                # Ask the question
                self.ask(question)
                print("-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

def main():
    """Main entry point"""
    print("\n🚀 Initializing AskMyCity RAG System...\n")
    
    # Create RAG system
    rag = AskMyCityRAG()
    
    # Load documents (only first time)
    rag.load_documents()
    
    # Start interactive chat
    rag.chat()

if __name__ == "__main__":
    main()