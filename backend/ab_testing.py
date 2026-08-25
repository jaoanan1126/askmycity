"""
A/B Testing Framework
Compare different RAG configurations
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class ABTest:
    """
    Compare two variants of RAG system
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.variant_a = {"name": "Control", "results": []}
        self.variant_b = {"name": "Treatment", "results": []}
    
    def add_result(self, variant: str, quality_score: float, hallucination_score: float):
        """Add a test result"""
        result = {
            "quality_score": quality_score,
            "hallucination_score": hallucination_score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if variant == "A":
            self.variant_a["results"].append(result)
        elif variant == "B":
            self.variant_b["results"].append(result)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze A/B test results
        
        Returns statistical significance of differences
        """
        a_quality = [r["quality_score"] for r in self.variant_a["results"]]
        b_quality = [r["quality_score"] for r in self.variant_b["results"]]
        
        a_hallucination = [r["hallucination_score"] for r in self.variant_a["results"]]
        b_hallucination = [r["hallucination_score"] for r in self.variant_b["results"]]
        
        # T-test for quality scores
        quality_ttest = stats.ttest_ind(a_quality, b_quality) if len(a_quality) > 1 and len(b_quality) > 1 else None
        
        # T-test for hallucination scores
        hallucination_ttest = stats.ttest_ind(a_hallucination, b_hallucination) if len(a_hallucination) > 1 and len(b_hallucination) > 1 else None
        
        return {
            "test_name": self.name,
            "description": self.description,
            "sample_size": {
                "variant_a": len(a_quality),
                "variant_b": len(b_quality)
            },
            "quality": {
                "variant_a_mean": round(np.mean(a_quality), 3) if a_quality else 0,
                "variant_b_mean": round(np.mean(b_quality), 3) if b_quality else 0,
                "difference": round(np.mean(b_quality) - np.mean(a_quality), 3) if a_quality and b_quality else 0,
                "p_value": round(quality_ttest.pvalue, 4) if quality_ttest else None,
                "significant": quality_ttest.pvalue < 0.05 if quality_ttest else False
            },
            "hallucination": {
                "variant_a_mean": round(np.mean(a_hallucination), 3) if a_hallucination else 0,
                "variant_b_mean": round(np.mean(b_hallucination), 3) if b_hallucination else 0,
                "difference": round(np.mean(b_hallucination) - np.mean(a_hallucination), 3) if a_hallucination and b_hallucination else 0,
                "p_value": round(hallucination_ttest.pvalue, 4) if hallucination_ttest else None,
                "significant": hallucination_ttest.pvalue < 0.05 if hallucination_ttest else False
            },
            "recommendation": self._get_recommendation(
                np.mean(b_quality) - np.mean(a_quality) if a_quality and b_quality else 0,
                quality_ttest.pvalue if quality_ttest else 1
            )
        }
    
    def _get_recommendation(self, quality_diff: float, p_value: float) -> str:
        """Get recommendation based on results"""
        if p_value > 0.05:
            return "No significant difference - more data needed"
        elif quality_diff > 0.05:
            return "✅ Variant B is significantly better - roll out!"
        elif quality_diff < -0.05:
            return "❌ Variant A is better - keep control"
        else:
            return "Difference is significant but small - consider cost/complexity"