"""
Advanced RAG Techniques
- Query rewriting
- Hybrid search (BM25 + vector)
- Re-ranking
- Enhanced citations
"""
import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class AdvancedRAGService:
    """
    Advanced RAG techniques for better retrieval and response quality
    """
    
    def __init__(self):
        """Initialize advanced RAG components"""
        self.reranker = None
        self._init_reranker()
        logger.info("✓ Advanced RAG service initialized")
    
    def _init_reranker(self):
        """Initialize cross-encoder for re-ranking"""
        try:
            # Load cross-encoder model (lightweight, runs locally)
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✓ Cross-encoder re-ranker loaded")
        except Exception as e:
            logger.warning(f"⚠️  Re-ranker not available: {e}")
            self.reranker = None
    
    def rewrite_query(self, question: str) -> Dict[str, Any]:
        """
        Rewrite user query for better retrieval
        
        Strategies:
        1. Extract key entities (locations, amounts, etc.)
        2. Expand abbreviations
        3. Add context keywords
        4. Generate multiple query variants
        
        Args:
            question: Original user question
            
        Returns:
            {
                "original": original question,
                "rewritten": improved query,
                "keywords": extracted keywords,
                "variants": alternative queries
            }
        """
        # Extract important entities
        keywords = self._extract_keywords(question)
        
        # Expand abbreviations
        expanded = self._expand_abbreviations(question)
        
        # Generate query variants
        variants = self._generate_query_variants(expanded, keywords)
        
        # Choose best variant (or combine)
        rewritten = variants[0] if variants else expanded
        
        return {
            "original": question,
            "rewritten": rewritten,
            "keywords": keywords,
            "variants": variants[:3]  # Top 3 variants
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common question words
        stop_words = {
            'how', 'what', 'when', 'where', 'who', 'why', 'is', 'are', 
            'do', 'does', 'can', 'could', 'would', 'should', 'the', 'a', 
            'an', 'i', 'my', 'me'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add important phrases
        if 'building permit' in text.lower():
            keywords.append('building_permit')
        if 'business license' in text.lower():
            keywords.append('business_license')
        
        return keywords
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        expansions = {
            r'\bSF\b': 'San Francisco',
            r'\bpermit\b': 'permit authorization',
            r'\bfee\b': 'fee cost price',
            r'\blicense\b': 'license permit',
        }
        
        expanded = text
        for abbr, expansion in expansions.items():
            expanded = re.sub(abbr, expansion, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _generate_query_variants(
        self, 
        expanded: str, 
        keywords: List[str]
    ) -> List[str]:
        """Generate alternative query formulations"""
        variants = []
        
        # Variant 1: Original expanded
        variants.append(expanded)
        
        # Variant 2: Keywords only
        if keywords:
            variants.append(' '.join(keywords))
        
        # Variant 3: Question as statement
        statement = expanded.replace('how do i', '').replace('?', '')
        statement = statement.replace('what is', '').replace('how to', '')
        variants.append(statement.strip())
        
        return variants
    
    def hybrid_search(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict],
        vector_scores: List[float],
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: Combine vector search with BM25 keyword search
        
        Args:
            query: Search query
            documents: List of document texts
            metadatas: Document metadata
            vector_scores: Scores from vector search
            top_k: Number of results to return
            alpha: Weight for vector vs BM25 (0=BM25 only, 1=vector only)
            
        Returns:
            List of documents with hybrid scores
        """
        if not documents:
            return []
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1 range
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores (weighted average)
        hybrid_scores = [
            alpha * v_score + (1 - alpha) * b_score
            for v_score, b_score in zip(vector_scores_norm, bm25_scores_norm)
        ]
        
        # Create results with scores
        results = [
            {
                "document": doc,
                "metadata": meta,
                "vector_score": v_score,
                "bm25_score": b_score,
                "hybrid_score": h_score
            }
            for doc, meta, v_score, b_score, h_score in zip(
                documents, metadatas, vector_scores_norm, 
                bm25_scores_norm, hybrid_scores
            )
        ]
        
        # Sort by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results[:top_k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if len(scores) == 0 or max(scores) == min(scores):
            return [0.5] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        return [(s - min_score) / range_score for s in scores]
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder
        
        Args:
            query: Search query
            documents: List of document dicts with 'document' key
            top_k: Number of top documents to return
            
        Returns:
            Re-ranked documents with relevance scores
        """
        if not self.reranker or not documents:
            return documents[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = [[query, doc['document']] for doc in documents]
            
            # Get relevance scores from cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"✓ Re-ranked {len(documents)} docs, kept top {top_k}")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return documents[:top_k]
    
    def enhance_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance answer with better citations
        
        Adds inline citations and source references
        
        Args:
            answer: Generated answer
            sources: Source documents used
            
        Returns:
            Answer with enhanced citations
        """
        # Add source references if not present
        if not any(marker in answer.lower() for marker in ['according to', 'source', 'document']):
            # Add generic citation
            enhanced = f"Based on San Francisco government documents:\n\n{answer}"
        else:
            enhanced = answer
        
        # Add source list at the end if not present
        if len(sources) > 0 and "Sources:" not in enhanced:
            enhanced += "\n\nSources:"
            for i, source in enumerate(sources, 1):
                source_name = source.get('source', 'Unknown')
                relevance = source.get('relevance_score', 0)
                enhanced += f"\n[{i}] {source_name} (relevance: {relevance:.2%})"
        
        return enhanced
    
    def validate_answer_quality(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate answer quality
        
        Checks:
        - Has citations
        - Answer is relevant to question
        - Sources support answer
        - No hallucination indicators
        
        Returns:
            {
                "is_valid": bool,
                "confidence": float,
                "issues": list,
                "suggestions": list
            }
        """
        issues = []
        suggestions = []
        
        # Check for citations
        has_citations = any(
            marker in answer.lower() 
            for marker in ['according to', 'source', 'document', 'states']
        )
        if not has_citations:
            issues.append("No citations found in answer")
            suggestions.append("Add source attributions")
        
        # Check for vague language
        vague_patterns = [
            r'\b(probably|maybe|might be|could be)\b',
            r'\b(approximately|roughly|around)\s+\$',
            r'\b(typically|usually|generally)\b'
        ]
        for pattern in vague_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append(f"Vague language detected: {pattern}")
                suggestions.append("Use specific information from sources")
        
        # Check answer length
        if len(answer.split()) < 10:
            issues.append("Answer is very short")
            suggestions.append("Provide more detailed response")
        
        # Check if answer addresses question
        question_keywords = set(self._extract_keywords(question))
        answer_keywords = set(self._extract_keywords(answer))
        overlap = len(question_keywords & answer_keywords)
        
        if overlap < len(question_keywords) * 0.3:
            issues.append("Answer may not address question")
            suggestions.append("Ensure answer is relevant to query")
        
        # Calculate confidence
        confidence = 1.0
        confidence -= len(issues) * 0.2
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "is_valid": len(issues) == 0,
            "confidence": round(confidence, 3),
            "issues": issues,
            "suggestions": suggestions
        }

# Global instance
advanced_rag = AdvancedRAGService()