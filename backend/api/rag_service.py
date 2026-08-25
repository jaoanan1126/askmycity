"""
RAG Service - Core business logic

WHY THIS FILE?
- Separates RAG logic from API logic
- Can test independently
- Can reuse in other contexts (CLI, notebooks, etc.)
"""
import anthropic
import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
import os
import time
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from .observability import observability
from .cost_tracker import cost_tracker
import sys
from pathlib import Path
# Add parent directory to path to import advanced_rag and evaluation
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_rag import AdvancedRAGService
from evaluation import EvaluationService

# Load environment variables from .env file
load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Configure logger
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.expanduser("~/Library/Application Support/askmycity/chroma_db")


class RAGService:
    """
    Encapsulates all RAG functionality

    This is your RAG system from Week 1, but organized as a class
    so FastAPI can use it easily!
    """

    def __init__(self):
        """Initialize Claude and ChromaDB"""
        print("🚀 Initializing RAG Service...")

        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ ANTHROPIC_API_KEY environment variable not set!\n"
                "Please either:\n"
                "  1. Create a .env file in the backend directory with: ANTHROPIC_API_KEY=your_key_here\n"
                "  2. Or export it in your terminal: export ANTHROPIC_API_KEY=your_key_here")

        # Initialize Claude
        self.claude = anthropic.Anthropic(api_key=api_key)
        print("✓ Claude initialized")

        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            self.collection = self.chroma_client.get_collection(
                name="sf_government",
                embedding_function=ONNXMiniLM_L6_V2())
            print(
                f"✓ Loaded {
                    self.collection.count()} documents from ChromaDB")
        except Exception as e:
            print(f"❌ Failed to load collection: {e}")
            raise

        # Store conversations in memory
        # In production, use Redis or database
        self.conversations: Dict[str, List[Dict]] = {}

        # Initialize Advanced RAG Service
        self.advanced_rag = AdvancedRAGService()

        # Track statistics
        self.stats = {
            "total_queries": 0,
            "total_response_time_ms": 0,
            "start_time": datetime.utcnow()
        }

        print("✓ RAG Service ready!\n")

    def get_uptime_seconds(self) -> float:
        """How long has the service been running?"""
        return (datetime.utcnow() - self.stats["start_time"]).total_seconds()

    def search_documents(self, query: str, n_results: int = 3) -> Dict:
        """
        Search ChromaDB for relevant documents

        This is the same semantic search you built in Week 1!
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a given ID"""
        return self.conversations.get(conversation_id, [])

    def save_conversation_turn(
        self,
        conversation_id: str,
        user_msg: str,
        assistant_msg: str
    ):
        """Save one question-answer pair to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])

        # Keep only last 10 messages (5 turns) to manage memory
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = \
                self.conversations[conversation_id][-10:]

    async def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        max_results: int = 3,
        use_advanced: bool = False  # 🆕 Enable advanced features
    ) -> Dict:
        """
        Main RAG function with advanced features

        New features:
        - Query rewriting for better retrieval
        - Hybrid search (vector + BM25)
        - Re-ranking for precision
        - Enhanced citations
        - Answer validation
        """
        start_time = time.time()

        # Ensure use_advanced is a boolean
        use_advanced = bool(use_advanced)

        # Generate conversation ID
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Create Langfuse trace
        trace = observability.create_trace(
            name="ask_question_advanced" if use_advanced else "ask_question",
            session_id=conversation_id,
            metadata={
                "question": question,
                "max_results": max_results,
                "advanced_features": use_advanced
            }
        )
        trace_id = trace.id if trace else None

        try:
            # 🆕 STEP 1: Query Rewriting (if advanced)
            if use_advanced:
                logger.info("🔄 Rewriting query for better retrieval...")
                query_info = self.advanced_rag.rewrite_query(question)
                search_query = query_info['rewritten']
                logger.info(f"📝 Rewritten: {search_query}")
            else:
                search_query = question

            # STEP 2: Initial retrieval (get more candidates for re-ranking)
            retrieve_count = max_results * 3 if use_advanced else max_results
            logger.info(f"🔍 Searching for {retrieve_count} documents...")

            search_results = self.search_documents(
                search_query, n_results=retrieve_count)

            # Log retrieval
            if trace_id:
                observability.log_retrieval(
                    trace_id=trace_id,
                    name="chromadb_search",
                    input_query=search_query,
                    output_documents=[
                        {
                            "source": meta['source'],
                            "category": meta['category'],
                            "relevance": round(1 - dist, 3)
                        }
            for meta, dist in zip(
                            search_results['metadatas'][0],
                            search_results['distances'][0]
                        )
                    ],
                    metadata={"n_results": retrieve_count, "advanced": use_advanced}
                )

            documents = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0]

            # 🆕 STEP 3: Hybrid Search (if advanced)
            if use_advanced and len(documents) > 0:
                logger.info("🔀 Applying hybrid search (vector + BM25)...")

                # Convert distances to similarity scores
                vector_scores = [1 - d for d in distances]

                # Hybrid search
                hybrid_results = self.advanced_rag.hybrid_search(
                    query=search_query,
                    documents=documents,
                    metadatas=metadatas,
                    vector_scores=vector_scores,
                    top_k=max_results * 2,
                    alpha=0.6  # 60% vector, 40% BM25
                )

                logger.info(
                    f"✓ Hybrid search returned {
                        len(hybrid_results)} candidates")

            else:
                # No hybrid search, use vector results
                hybrid_results = [
                    {
                        "document": doc,
                        "metadata": meta,
                        "vector_score": 1 - dist,
                        "hybrid_score": 1 - dist
                    }
                for doc, meta, dist in zip(documents, metadatas, distances)
                ][:max_results * 2]

            # 🆕 STEP 4: Re-ranking (if advanced)
            if use_advanced and len(hybrid_results) > 0:
                logger.info("🎯 Re-ranking documents with cross-encoder...")

                final_results = self.advanced_rag.rerank_documents(
                    query=question,  # Use original question for re-ranking
                    documents=hybrid_results,
                    top_k=max_results
                )

                logger.info(
                    f"✓ Selected top {
                        len(final_results)} documents after re-ranking")
            else:
                final_results = hybrid_results[:max_results]

            # STEP 5: Build context
            context = ""
            sources = []

            for i, result in enumerate(final_results, 1):
                doc = result['document']
                meta = result['metadata']

                context += f"[Document {i}]\n{doc}\n\n"
                sources.append({
                    "source": meta['source'],
                    "category": meta['category'],
                    "chunk": meta['chunk'],
                    "relevance_score": round(result.get('rerank_score', result.get('hybrid_score', 0.5)), 3)
                })

            logger.info(f"📚 Built context from {len(sources)} sources")

            # STEP 6: Get conversation history
            history = self.get_conversation_history(conversation_id)

            # 🆕 STEP 7: Enhanced system prompt (if advanced)
            if use_advanced:
                system_prompt = """You are AskMyCity, a helpful assistant for San Francisco government services.

            CRITICAL INSTRUCTIONS FOR HIGH-QUALITY RESPONSES:
            1. Answer ONLY using information from the provided documents
            2. ALWAYS cite your sources: "According to [Document X], ..."
            3. Be specific with numbers, addresses, fees, and contact information
            4. If information is not in the documents, clearly state: "I don't have that information in my knowledge base."
            5. NEVER invent or guess:
            - Phone numbers or addresses
            - Fees or costs
            - Dates or deadlines
            - Policies or procedures
            6. If sources conflict, mention both versions
            7. Use exact quotes for important facts
            8. Maintain conversation context from previous messages

            RESPONSE FORMAT:
            - Start with a direct answer
            - Cite which document(s) support each claim
            - Be concise but complete
            - End with source references if helpful

            QUALITY STANDARDS:
            - Every factual claim must be traceable to a source
            - Use "The source states..." not "I know..."
            - If uncertain, acknowledge uncertainty
            - Prioritize accuracy over completeness"""
            else:
                system_prompt = """You are AskMyCity, a helpful assistant for San Francisco government services.

            Answer questions based ONLY on the provided context documents.
            Be specific and cite sources when possible."""

            # STEP 8: Build messages
            messages = history[-8:] if history else []

            current_message = f"""Context documents:

            {context}

            User question: {question}

            Please provide a clear, helpful answer based on the context above."""

            messages.append({
                "role": "user",
                "content": current_message
            })

            # STEP 9: Generate response
            logger.info("🤖 Generating response with Claude...")
            response = self.claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )

            full_response = response.content[0].text

            # Log generation
            if trace_id:
                observability.log_generation(
                    trace_id=trace_id,
                    name="claude_generation_advanced" if use_advanced else "claude_generation",
                    model="claude-haiku-4-5-20251001",
                    input_text=current_message[:1000] + "..." if len(current_message) > 1000 else current_message,
                    output_text=full_response,
                    usage={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens
                    },
                    metadata={
                        "advanced_features": use_advanced,
                        "sources_count": len(sources),
                        "context_length": len(context)
                    }
                )

            logger.info(
                f"✅ Generated {
                    response.usage.output_tokens} output tokens")

            # 🆕 STEP 10: Enhance citations (if advanced)
            if use_advanced:
                logger.info("📎 Enhancing citations...")
                full_response = self.advanced_rag.enhance_citations(
                    full_response, sources)

            # 🆕 STEP 11: Validate answer quality (if advanced)
            validation = None
            if use_advanced:
                logger.info("✅ Validating answer quality...")
                validation = self.advanced_rag.validate_answer_quality(
                    question=question,
                    answer=full_response,
                    sources=sources
                )

                if not validation['is_valid']:
                    logger.warning(f"⚠️  Quality issues: {validation['issues']}")

            # STEP 12: Save conversation
            self.save_conversation_turn(
                conversation_id, question, full_response)

            # STEP 13: Track costs
            cost_tracker.track_request(
                conversation_id=conversation_id,
                model="claude-haiku-4-5-20251001",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached=False,
                metadata={
                    "advanced_features": use_advanced,
                    "sources_count": len(sources)
                }
            )

            # STEP 14: Calculate metrics
            response_time_ms = (time.time() - start_time) * 1000

            # STEP 15: Update stats
            self.stats["total_queries"] += 1
            self.stats["total_response_time_ms"] += response_time_ms

            # STEP 16: Return result
            result = {
                "question": question,
                "answer": full_response,
                "sources": sources,
                "conversation_id": conversation_id,
                "response_time_ms": round(
                    response_time_ms,
                    2),
                "trace_id": trace_id,
                "tokens": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "total": response.usage.input_tokens +
                        response.usage.output_tokens}}

            # Add advanced features metadata
            if use_advanced and validation:
                result["validation"] = validation
                result["advanced_features_used"] = [
                    "query_rewriting",
                    "hybrid_search",
                    "reranking",
                    "enhanced_citations",
                    "answer_validation"
                ]

            logger.info(f"⏱️  Total response time: {response_time_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline: {e}", exc_info=True)

            if trace_id:
                observability.log_score(
                    trace_id=trace_id,
                    name="error",
                    value=0,
                    comment=str(e)
                )

            raise

        finally:
            observability.flush()

    def get_stats(self) -> Dict:
        """Get service statistics"""
        avg_response_time = 0
        if self.stats["total_queries"] > 0:
            avg_response_time = \
                self.stats["total_response_time_ms"] / self.stats["total_queries"]

        return {
            "total_queries": self.stats["total_queries"],
            "total_conversations": len(self.conversations),
            "average_response_time_ms": round(avg_response_time, 2),
            "documents_in_db": self.collection.count(),
        }

    def get_health(self) -> Dict:
        """Health check - is everything working?"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "documents_loaded": self.collection.count(),
            "uptime_seconds": round(self.get_uptime_seconds(), 2)
        }
