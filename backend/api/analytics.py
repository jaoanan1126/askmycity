"""
Analytics service to track usage
"""
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AnalyticsService:
    """
    Track queries, response times, and user feedback
    """
    
    def __init__(self, data_file: str = "analytics.json"):
        """Initialize analytics service"""
        self.data_file = Path(data_file)
        self.data = self._load_data()
        logger.info("✓ Analytics service initialized")
    
    def _load_data(self) -> Dict[str, Any]:
        """Load analytics data from file"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        
        return {
            "queries": [],
            "feedback": [],
            "popular_questions": defaultdict(int)
        }
    
    def _save_data(self) -> None:
        """Save analytics data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def track_query(
        self,
        question: str,
        response_time_ms: float,
        sources_count: int,
        conversation_id: str,
        cached: bool = False
    ) -> None:
        """Track a query"""
        self.data["queries"].append({
            "question": question,
            "response_time_ms": response_time_ms,
            "sources_count": sources_count,
            "conversation_id": conversation_id,
            "cached": cached,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Track popular questions (normalized)
        normalized = question.lower().strip()
        self.data["popular_questions"][normalized] = \
            self.data["popular_questions"].get(normalized, 0) + 1
        
        self._save_data()
        logger.info(f"✓ Tracked query: {question[:50]}...")
    
    def track_feedback(
        self,
        conversation_id: str,
        rating: int,
        comment: str = None
    ) -> None:
        """Track user feedback"""
        self.data["feedback"].append({
            "conversation_id": conversation_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self._save_data()
        logger.info(f"✓ Tracked feedback: {rating}/5")
    
    def get_popular_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular questions"""
        sorted_questions = sorted(
            self.data["popular_questions"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"question": q, "count": c}
            for q, c in sorted_questions[:limit]
        ]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        queries = self.data["queries"]
        feedback = self.data["feedback"]
        
        # Calculate averages
        avg_response_time = 0
        if queries:
            avg_response_time = sum(q["response_time_ms"] for q in queries) / len(queries)
        
        avg_rating = 0
        if feedback:
            avg_rating = sum(f["rating"] for f in feedback) / len(feedback)
        
        return {
            "total_queries": len(queries),
            "average_response_time_ms": round(avg_response_time, 2),
            "total_feedback": len(feedback),
            "average_rating": round(avg_rating, 2),
            "popular_questions": self.get_popular_questions(5)
        }

# Global analytics instance
analytics_service = AnalyticsService()