"""
Pydantic models for request/response validation

WHY THIS FILE?
- Defines the "shape" of data
- Automatic validation
- Automatic documentation
- Type safety
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

class AskRequest(BaseModel):
    """
    What the user sends when asking a question
    
    Example:
    {
        "question": "How do I get a permit?",
        "conversation_id": "conv_123",  # optional
        "max_results": 3
    }
    """
    question: str = Field(
        ...,  # Required field
        min_length=1,
        max_length=500,
        description="User's question"
    )
    conversation_id: Optional[str] = Field(
        None,  # Optional field
        description="Optional conversation ID for context"
    )
    max_results: Optional[int] = Field(
        3,  # Default value
        ge=1,  # Greater than or equal to 1
        le=10,  # Less than or equal to 10
        description="Number of documents to retrieve"
    )
    
    class Config:
        # Example for API docs
        json_schema_extra = {
            "example": {
                "question": "How do I get a building permit?",
                "conversation_id": "conv_123",
                "max_results": 3
            }
        }

class Source(BaseModel):
    """Metadata about a source document"""
    source: str         # e.g., "building_permits"
    category: str       # e.g., "permits"
    chunk: int          # e.g., 0 (which chunk)
    relevance_score: Optional[float] = None  # How relevant (0-1)

class AskResponse(BaseModel):
    """
    What the API sends back to the user
    
    Example:
    {
        "question": "How do I get a permit?",
        "answer": "To apply for a building permit...",
        "sources": [{"source": "building_permits", ...}],
        "conversation_id": "conv_123",
        "response_time_ms": 1234.56
    }
    """
    question: str
    answer: str
    sources: List[Source]
    conversation_id: str
    response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Optional extras (dropped silently by FastAPI unless declared here)
    trace_id: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None
    cached: bool = False
    validation: Optional[Dict[str, Any]] = None
    advanced_features_used: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str              # "healthy" or "unhealthy"
    version: str             # "1.0.0"
    documents_loaded: int    # How many docs in ChromaDB
    uptime_seconds: float    # How long server has been running

class StatsResponse(BaseModel):
    """Statistics response"""
    total_queries: int
    total_conversations: int
    average_response_time_ms: float
    documents_in_db: int
    cache: Dict[str, Any]       # SemanticCache.get_stats()
    analytics: Dict[str, Any]   # AnalyticsService.get_analytics_summary() (all-time, persisted)

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FeedbackRequest(BaseModel):
    """Feedback request"""
    conversation_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    comment: Optional[str] = Field(None, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "rating": 5,
                "comment": "Very helpful!"
            }
        }