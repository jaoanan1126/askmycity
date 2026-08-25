"""
Smart cache with semantic similarity matching
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Cache that matches semantically similar questions
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.85
    ):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0
        }
        
        logger.info(f"✓ Semantic cache initialized (threshold: {similarity_threshold})")
    
    def _normalize(self, text: str) -> str:
        """Basic normalization"""
        return text.lower().strip()
    
    def _find_similar(self, question: str) -> Optional[Tuple[str, float]]:
        """Find most similar cached question"""
        if not self.embeddings:
            return None
        
        # Get embedding for query
        query_embedding = self.model.encode(question)
        
        best_match = None
        best_score = 0.0
        
        # Compare with all cached embeddings
        for cached_q, cached_emb in self.embeddings.items():
            # Cosine similarity
            similarity = float(np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            ))
            
            if similarity > best_score:
                best_score = similarity
                best_match = cached_q
        
        if best_score >= self.similarity_threshold:
            return (best_match, best_score)
        
        return None
    
    def get(self, question: str, max_results: int = 3) -> Optional[Dict[str, Any]]:
        """Get cached response (exact or semantic match)"""
        normalized = self._normalize(question)
        cache_key = f"{normalized}:{max_results}"
        
        # Try exact match first
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.utcnow() < cached["expires_at"]:
                self.stats["exact_hits"] += 1
                logger.info(f"⚡ Exact cache hit")
                return cached["data"]
            else:
                del self.cache[cache_key]
        
        # Try semantic match
        similar = self._find_similar(question)
        if similar:
            cached_q, score = similar
            cached_key = f"{cached_q}:{max_results}"
            
            if cached_key in self.cache:
                cached = self.cache[cached_key]
                if datetime.utcnow() < cached["expires_at"]:
                    self.stats["semantic_hits"] += 1
                    logger.info(f"🎯 Semantic cache hit (similarity: {score:.2%})")
                    return cached["data"]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, question: str, response: Dict[str, Any], max_results: int = 3) -> None:
        """Cache response with embedding"""
        normalized = self._normalize(question)
        cache_key = f"{normalized}:{max_results}"
        
        # Store response
        self.cache[cache_key] = {
            "data": response,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        }
        
        # Store embedding
        self.embeddings[normalized] = self.model.encode(question)
        
        logger.info(f"✓ Cached with embedding")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = sum(self.stats.values())
        hit_rate = 0.0
        if total > 0:
            hit_rate = ((self.stats["exact_hits"] + self.stats["semantic_hits"]) / total) * 100
        
        return {
            "exact_hits": self.stats["exact_hits"],
            "semantic_hits": self.stats["semantic_hits"],
            "cache_misses": self.stats["misses"],
            "semantic_hit_rate": round(hit_rate, 2),
            "cached_items": len(self.cache)
        }
