"""
FastAPI Application - AskMyCity API
"""
from .cache import SemanticCache
from .models import AskRequest, AskResponse, FeedbackRequest, HealthResponse, StatsResponse
from .rag_service import RAGService 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .analytics import analytics_service
from .logging_config import setup_logging, set_request_context, clear_request_context, log_with_context
import asyncio
import uuid
import os
import time
from .observability import observability
import sys
from pathlib import Path
# Add parent directory to path to import evaluation and advanced_rag
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation import evaluation_service
from advanced_rag import AdvancedRAGService




# Configure logging
JSON_LOGS = os.getenv("JSON_LOGS", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level=LOG_LEVEL, json_logs=JSON_LOGS)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AskMyCity API",
    description="AI-powered San Francisco government assistant using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG service
rag_service = None
cache = None
advanced_rag = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests with context
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Set request context
    set_request_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        ip_address=request.client.host if request.client else "unknown"
    )
    
    # Log request start
    log_with_context(
        logger,
        "info",
        f"Request started: {request.method} {request.url.path}",
        request_id=request_id,
        method=request.method,
        path=request.url.path
    )
    
    # Process request
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request completion
        log_with_context(
            logger,
            "info",
            f"Request completed: {request.method} {request.url.path}",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Log error
        log_with_context(
            logger,
            "error",
            f"Request failed: {request.method} {request.url.path}",
            request_id=request_id,
            error=str(e),
            duration_ms=round(duration_ms, 2)
        )
        
        raise
    
    finally:
        # Clear context
        clear_request_context()

# ============================================
# ADVANCED RAG ENDPOINTS
# ============================================

@app.post("/api/ask-advanced", response_model=AskResponse, tags=["RAG"])
@limiter.limit("20/minute")
async def ask_question_advanced(request: Request, ask_request: AskRequest):
    """
    Ask a question with ADVANCED RAG features
    
    Advanced features:
    - Query rewriting for better retrieval
    - Hybrid search (vector + BM25)
    - Cross-encoder re-ranking
    - Enhanced citations
    - Answer validation
    
    Use this for highest quality answers!
    """
    # Same as ask_question but with use_advanced=True
    try:
        set_request_context(
            conversation_id=ask_request.conversation_id,
            question_length=len(ask_request.question),
            max_results=ask_request.max_results,
            advanced=True
        )
        
        log_with_context(
            logger,
            "info",
            "Advanced question received",
            question_preview=ask_request.question[:100]
        )
        
        if not ask_request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Call RAG with advanced features
        result = await rag_service.ask(
            question=ask_request.question,
            conversation_id=ask_request.conversation_id,
            max_results=ask_request.max_results,
            use_advanced=True  # 🆕 Enable advanced features
        )
        
        log_with_context(
            logger,
            "info",
            "Advanced response generated",
            response_time_ms=result['response_time_ms'],
            validation_confidence=result.get('validation', {}).get('confidence', 1.0)
        )
        
        return AskResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_with_context(logger, "error", "Error in advanced RAG", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/query-rewrite", tags=["Advanced RAG"])
async def rewrite_query(question: str):
    """
    Test query rewriting
    
    Shows how a query is transformed for better retrieval
    
    Example: /api/query-rewrite?question=How%20much%20for%20a%20permit?
    """
    try:
        result = advanced_rag.rewrite_query(question)
        return result
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global rag_service, cache, advanced_rag
    logger.info("="*70)
    logger.info("🚀 Starting AskMyCity API...")
    logger.info("="*70)
    
    try:
        rag_service = RAGService()
        cache = SemanticCache()
        advanced_rag = AdvancedRAGService()
        logger.info("="*70)
        logger.info("✅ AskMyCity API is ready!")
        logger.info(f"📚 Loaded {rag_service.collection.count()} documents")
        logger.info("📖 API docs available at: http://localhost:8000/docs")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down AskMyCity API...")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================
# ROUTES
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Welcome to AskMyCity API",
        "version": "1.0.0",
        "description": "AI-powered SF government assistant",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "ask": "POST /api/ask"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("20/minute") 
async def health_check(request: Request):
    """
    Health check endpoint
    
    Returns system health status
    """
    try:
        health = rag_service.get_health()
        return health
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """
    Get service statistics
    
    Returns usage metrics
    """
    try:
        # Top level: this server process (in-memory, since startup).
        # analytics: all-time numbers persisted in analytics.json.
        # Nested so overlapping keys (total_queries, average_response_time_ms)
        # don't overwrite each other.
        return {
            **rag_service.get_stats(),
            "cache": cache.get_stats(),
            "analytics": analytics_service.get_analytics_summary(),
        }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve stats"
        )

# ============================================
# EVALUATION ENDPOINTS
# ============================================

@app.post("/api/evaluate", tags=["Evaluation"])
@limiter.limit("5/hour")
async def run_evaluation(request: Request, use_ragas: bool = False):
    """
    Run full system evaluation
    
    Tests RAG system against test set and provides:
    - Quality scores
    - Hallucination detection
    - Performance metrics
    - RAGAS scores (optional)
    
    Args:
        use_ragas: Use RAGAS framework for advanced metrics
    
    Rate limited to 5 per hour due to computational cost
    """
    try:
        logger.info("🧪 Starting evaluation...")
        
        results = await evaluation_service.evaluate_system(
            rag_service=rag_service,
            use_ragas=use_ragas,
            save_results=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

@app.get("/api/evaluate/history", tags=["Evaluation"])
async def get_evaluation_history():
    """
    Get historical evaluation results
    
    Returns all past evaluation runs for tracking improvements
    """
    try:
        return {
            "evaluations": evaluation_service.results_history,
            "total_runs": len(evaluation_service.results_history)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve evaluation history"
        )

@app.get("/api/evaluate/trends", tags=["Evaluation"])
async def get_evaluation_trends():
    """
    Get evaluation trends over time
    
    Shows how quality metrics have changed across evaluation runs
    """
    try:
        history = evaluation_service.results_history
        
        if not history:
            return {"message": "No evaluation history yet"}
        
        trends = {
            "timestamps": [],
            "quality_scores": [],
            "hallucination_rates": [],
            "topic_coverage": []
        }
        
        for result in history:
            trends["timestamps"].append(result.get('timestamp'))
            
            metrics = result.get('metrics', {})
            trends["quality_scores"].append(metrics.get('average_quality_score', 0))
            trends["topic_coverage"].append(metrics.get('average_topic_coverage', 0))
            
            hallucination = result.get('hallucination_analysis', {})
            trends["hallucination_rates"].append(hallucination.get('hallucination_rate', 0))
        
        return trends
        
    except Exception as e:
        logger.error(f"❌ Failed to get trends: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve trends"
        )
    
@app.post("/api/ask", response_model=AskResponse, tags=["RAG"])
@limiter.limit("20/minute")
async def ask_question(request: Request, ask_request: AskRequest):
    """
    Ask a question to AskMyCity
    
    Features:
    - Rate limiting (20/min)
    - Caching (smart normalization)
    - Analytics tracking
    - Structured logging
    - Langfuse observability
    - Cost tracking
    """
    try:
        # Set conversation context for logging
        set_request_context(
            conversation_id=ask_request.conversation_id,
            question_length=len(ask_request.question),
            max_results=ask_request.max_results
        )
        
        log_with_context(
            logger,
            "info",
            "Question received",
            question_preview=ask_request.question[:100],
            conversation_id=ask_request.conversation_id,
            max_results=ask_request.max_results
        )
        
        # Validate question
        if not ask_request.question.strip():
            log_with_context(logger, "warning", "Empty question rejected")
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Only use the cache for the FIRST question of a conversation.
        # Follow-up questions depend on conversation history, so a cached
        # answer to a similar standalone question could be wrong for them.
        use_cache = not rag_service.get_conversation_history(
            ask_request.conversation_id or "")

        if use_cache:
            lookup_start = time.time()
            cached_response = await asyncio.to_thread(
                cache.get, ask_request.question, ask_request.max_results)

            if cached_response:
                # Re-bind the cached answer to THIS request: its own
                # question, conversation ID and timing (never another
                # user's conversation_id).
                conversation_id = (
                    ask_request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}")
                cached_response.update(
                    question=ask_request.question,
                    conversation_id=conversation_id,
                    response_time_ms=round((time.time() - lookup_start) * 1000, 2),
                    trace_id=None,
                    tokens=None,  # no tokens spent on a cache hit
                )
                # Record the turn so follow-up questions have context
                rag_service.save_conversation_turn(
                    conversation_id, ask_request.question, cached_response["answer"])

                log_with_context(
                    logger,
                    "info",
                    "Cache hit - returning cached response",
                    cache_hit=True,
                    response_time_ms=cached_response["response_time_ms"]
                )

                # Track analytics (cached)
                if 'analytics_service' in globals():
                    analytics_service.track_query(
                        question=ask_request.question,
                        response_time_ms=cached_response["response_time_ms"],
                        sources_count=len(cached_response["sources"]),
                        conversation_id=conversation_id,
                        cached=True
                    )

                return AskResponse(**cached_response)

        # Cache miss - call RAG service
        log_with_context(
            logger,
            "info",
            "Cache miss - calling RAG service",
            cache_hit=False
        )
        
        # Get answer from RAG service
        # This now includes:
        # - Langfuse tracking
        # - Cost tracking
        # - Token counting
        result = await rag_service.ask(
            question=ask_request.question,
            conversation_id=ask_request.conversation_id,
            max_results=ask_request.max_results
        )
        
        # Cache standalone (first-turn) answers only
        if use_cache:
            await asyncio.to_thread(
                cache.set, ask_request.question, result, ask_request.max_results)

        # Track analytics (if analytics_service exists)
        if 'analytics_service' in globals():
            analytics_service.track_query(
                question=ask_request.question,
                response_time_ms=result["response_time_ms"],
                sources_count=len(result["sources"]),
                conversation_id=result["conversation_id"],
                cached=False
            )
        
        log_with_context(
            logger,
            "info",
            "Response generated successfully",
            response_time_ms=result['response_time_ms'],
            sources_count=len(result['sources']),
            tokens_used=result.get('tokens', {}).get('total', 0),
            trace_id=result.get('trace_id'),
            cached=False
        )
        
        return AskResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_with_context(
            logger,
            "error",
            "Error processing question",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )
        
@app.get("/api/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: str):
    """
    Get conversation history
    
    Returns all messages in a conversation
    """
    try:
        history = rag_service.get_conversation_history(conversation_id)
        
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found"
            )
        
        return {
            "conversation_id": conversation_id,
            "messages": history,
            "message_count": len(history),
            "turns": len(history) // 2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation"
        )

@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback
    
    - **conversation_id**: The conversation to rate
    - **rating**: 1-5 stars
    - **comment**: Optional text feedback
    """
    try:
        analytics_service.track_feedback(
            conversation_id=feedback.conversation_id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"❌ Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback"
        )

# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
