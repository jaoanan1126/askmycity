"""
LLM Evaluation Framework
Measure RAG quality, detect hallucinations, track improvements
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

# Try to import RAGAS (optional)
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not installed - advanced metrics unavailable")

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Evaluate RAG system quality
    """
    
    def __init__(self, test_file: str = "test_questions.json"):
        """Initialize evaluation service"""
        self.test_file = Path(test_file)
        self.test_cases = self._load_test_cases()
        self.results_file = Path("evaluation_results.json")
        self.results_history = self._load_results_history()
        logger.info(f"✓ Evaluation service initialized with {len(self.test_cases)} test cases")
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from file"""
        if not self.test_file.exists():
            logger.warning(f"Test file not found: {self.test_file}")
            return []
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
            return data.get('test_cases', [])
    
    def _load_results_history(self) -> List[Dict[str, Any]]:
        """Load evaluation history"""
        if not self.results_file.exists():
            return []
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results"""
        self.results_history.append(results)
        
        with open(self.results_file, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
    
    async def evaluate_system(
        self,
        rag_service,
        use_ragas: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run full evaluation on RAG system
        
        Args:
            rag_service: RAG service instance
            use_ragas: Use RAGAS metrics (requires installation)
            save_results: Save results to history
        
        Returns:
            Complete evaluation results
        """
        logger.info("="*70)
        logger.info("🧪 Starting RAG System Evaluation")
        logger.info("="*70)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_questions": len(self.test_cases),
            "responses": [],
            "metrics": {},
            "hallucination_analysis": {},
            "performance": {}
        }
        
        # Run each test case
        total_time = 0
        for i, test_case in enumerate[Dict[str, Any]](self.test_cases, 1):
            logger.info(f"\n[{i}/{len(self.test_cases)}] Testing: {test_case['question'][:50]}...")
            
            try:
                # Get response from RAG system
                response = await rag_service.ask(
                    question=test_case['question'],
                    max_results=3
                )
                
                # print("RESPONSE: ", response)
                # Analyze response
                analysis = self._analyze_response(
                    test_case=test_case,
                    response=response
                )
                
                results["responses"].append(analysis)
                total_time += response['response_time_ms']
                
                logger.info(f"✓ Completed - Score: {analysis['quality_score']:.2f}")
                
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                results["responses"].append({
                    "question": test_case['question'],
                    "error": str(e),
                    "quality_score": 0
                })
        
        # Calculate aggregate metrics
        results["metrics"] = self._calculate_metrics(results["responses"])
        results["hallucination_analysis"] = self._analyze_hallucinations(results["responses"])
        results["performance"] = {
            "avg_response_time_ms": total_time / len(self.test_cases) if self.test_cases else 0,
            "total_time_ms": total_time
        }
        
        # Use RAGAS if available
        if use_ragas and RAGAS_AVAILABLE:
            logger.info("\n📊 Running RAGAS evaluation...")
            ragas_results = await self._evaluate_with_ragas(results["responses"])
            results["ragas_metrics"] = ragas_results
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        if save_results:
            self._save_results(results)
            logger.info(f"\n💾 Results saved to {self.results_file}")
        
        logger.info("="*70)
        logger.info("✅ Evaluation Complete!")
        logger.info("="*70)
        
        return results
    
    def _analyze_response(
        self,
        test_case: Dict[str, Any],
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single response
        
        Checks:
        - Answer quality
        - Topic coverage
        - Hallucination indicators
        - Source quality
        """
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        contexts = response.get('contexts', [])
        
        # Check if answer addresses expected topics
        expected_topics = test_case.get('expected_topics', [])
        topics_covered = [
            topic for topic in expected_topics
            if topic.lower() in answer.lower()
        ]
        topic_coverage = len(topics_covered) / len(expected_topics) if expected_topics else 0
        
        # Check for hallucination indicators
        hallucination_score = self._check_hallucination_indicators(answer, contexts)
        
        # Check source quality
        source_quality = self._assess_source_quality(sources)
        
        # Calculate overall quality score
        quality_score = (
            topic_coverage * 0.4 +
            (1 - hallucination_score) * 0.4 +
            source_quality * 0.2
        )
        
        return {
            "question": test_case['question'],
            "answer": answer,
            "ground_truth": test_case.get('ground_truth'),
            "expected_topics": expected_topics,
            "topics_covered": topics_covered,
            "topic_coverage": round(topic_coverage, 3),
            "hallucination_score": round(hallucination_score, 3),
            "source_quality": round(source_quality, 3),
            "quality_score": round(quality_score, 3),
            "sources_count": len(sources),
            "contexts": contexts,
            "response_time_ms": response.get('response_time_ms', 0)
        }
    
    def _check_hallucination_indicators(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Detect hallucination indicators
        
        Args:
            contexts: text of the retrieved documents the answer was based on
        
        Returns:
            Score 0-1 (0 = no hallucination, 1 = likely hallucination)
        """
        import re
        
        hallucination_patterns = [
            # Suspicious phrases
            (r"(?i)(i think|i believe|probably|maybe|might be)", 0.3),
            (r"(?i)(approximately|roughly|around) \$[\d,]+", 0.2),
            (r"(?i)(typically|usually|generally|often) (?!according to)", 0.2),
            
            # Specific claims without citation
            (r"\$[\d,]+\.\d{2}", 0.4),  # Exact dollar amounts
            (r"\d{3}-\d{3}-\d{4}", 0.5),  # Phone numbers
            (r"\d+ [A-Z][a-z]+ (Street|Avenue|Road|Boulevard)", 0.4),  # Addresses
        ]
        
        hallucination_score = 0.0
        
        for pattern, weight in hallucination_patterns:
            # finditer + group(0) gives the full match ("100 Larkin Street");
            # findall would return only the capture group ("Street").
            matches = [m.group(0) for m in re.finditer(pattern, answer)]
            if matches:
                # Check if matched text appears in the retrieved documents
                found_in_sources = False
                for match in matches:
                    for source_text in contexts:
                        if match in source_text:
                            found_in_sources = True
                            break
                    if found_in_sources:
                        break
                
                if not found_in_sources:
                    hallucination_score += weight
        
        return min(hallucination_score, 1.0)
    
    def _assess_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """
        Assess quality of retrieved sources
        
        Returns:
            Score 0-1 (0 = poor, 1 = excellent)
        """
        if not sources:
            return 0.0
        
        # Check relevance scores
        relevance_scores = [s.get('relevance_score', 0) for s in sources]
        avg_relevance = np.mean(relevance_scores)
        
        # Check if we have enough sources
        source_count_score = min(len(sources) / 3, 1.0)  # Ideal: 3+ sources
        
        # Combined score
        quality = (avg_relevance * 0.7 + source_count_score * 0.3)
        
        return quality
    
    def _calculate_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        if not responses:
            return {}
        
        # Filter out errors
        valid_responses = [r for r in responses if 'error' not in r]
        
        if not valid_responses:
            return {"error": "No valid responses"}
        
        return {
            "average_quality_score": round(np.mean([r['quality_score'] for r in valid_responses]), 3),
            "average_topic_coverage": round(np.mean([r['topic_coverage'] for r in valid_responses]), 3),
            "average_hallucination_score": round(np.mean([r['hallucination_score'] for r in valid_responses]), 3),
            "average_source_quality": round(np.mean([r['source_quality'] for r in valid_responses]), 3),
            "questions_with_high_quality": sum(1 for r in valid_responses if r['quality_score'] > 0.7),
            "questions_with_hallucinations": sum(1 for r in valid_responses if r['hallucination_score'] > 0.3),
            "success_rate": round(len(valid_responses) / len(responses), 3)
        }
    
    def _analyze_hallucinations(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detailed hallucination analysis"""
        valid_responses = [r for r in responses if 'error' not in r]
        
        if not valid_responses:
            return {}
        
        hallucination_scores = [r['hallucination_score'] for r in valid_responses]
        
        return {
            "total_responses": len(valid_responses),
            "responses_with_hallucinations": sum(1 for s in hallucination_scores if s > 0.3),
            "hallucination_rate": round(sum(1 for s in hallucination_scores if s > 0.3) / len(hallucination_scores), 3),
            "average_hallucination_score": round(np.mean(hallucination_scores), 3),
            "max_hallucination_score": round(max(hallucination_scores), 3),
            "low_risk": sum(1 for s in hallucination_scores if s < 0.2),
            "medium_risk": sum(1 for s in hallucination_scores if 0.2 <= s < 0.5),
            "high_risk": sum(1 for s in hallucination_scores if s >= 0.5)
        }
    
    async def _evaluate_with_ragas(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate using RAGAS framework
        
        Note: Requires RAGAS installation and Anthropic API key
        """
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not installed"}
        
        try:
            # Prepare data for RAGAS
            data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": []
            }
            
            for response in responses:
                if 'error' in response:
                    continue
                
                data["question"].append(response['question'])
                data["answer"].append(response['answer'])
                data["contexts"].append(response.get('contexts', []))  # retrieved chunk texts
                data["ground_truth"].append(response.get('ground_truth', ''))
            
            # Create dataset
            dataset = Dataset.from_dict(data)
            
            # Evaluate
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                ]
            )
            
            return {
                "faithfulness": round(result['faithfulness'], 3),
                "answer_relevancy": round(result['answer_relevancy'], 3),
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        metrics = results.get('metrics', {})
        hallucination = results.get('hallucination_analysis', {})
        performance = results.get('performance', {})
        
        print(f"\n📈 Overall Quality:")
        print(f"  Average Quality Score: {metrics.get('average_quality_score', 0):.3f}")
        print(f"  Topic Coverage: {metrics.get('average_topic_coverage', 0):.3f}")
        print(f"  Source Quality: {metrics.get('average_source_quality', 0):.3f}")
        
        print(f"\n🎯 Hallucination Analysis:")
        print(f"  Hallucination Rate: {hallucination.get('hallucination_rate', 0):.1%}")
        print(f"  Responses with Hallucinations: {hallucination.get('responses_with_hallucinations', 0)}")
        print(f"  Low Risk: {hallucination.get('low_risk', 0)}")
        print(f"  Medium Risk: {hallucination.get('medium_risk', 0)}")
        print(f"  High Risk: {hallucination.get('high_risk', 0)}")
        
        print(f"\n⚡ Performance:")
        print(f"  Avg Response Time: {performance.get('avg_response_time_ms', 0):.0f}ms")
        
        if 'ragas_metrics' in results:
            ragas = results['ragas_metrics']
            if 'error' not in ragas:
                print(f"\n📊 RAGAS Metrics:")
                print(f"  Faithfulness: {ragas.get('faithfulness', 0):.3f}")
                print(f"  Answer Relevancy: {ragas.get('answer_relevancy', 0):.3f}")

# Global evaluation service
evaluation_service = EvaluationService()