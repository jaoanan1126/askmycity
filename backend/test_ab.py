"""
Example A/B test: Standard vs Anti-Hallucination Prompt
"""
import asyncio
from api.rag_service import RAGService
from evaluation import EvaluationService
from ab_testing import ABTest

async def run_ab_test():
    """Run A/B test comparing two prompts"""
    
    # Create test
    test = ABTest(
        name="Standard vs Anti-Hallucination Prompt",
        description="Compare current prompt with stricter anti-hallucination instructions"
    )
    
    # Load services
    rag_service = RAGService()
    eval_service = EvaluationService()
    
    # Test questions
    test_questions = eval_service.test_cases[:5]  # Use first 5 questions
    
    print("🧪 Running A/B Test...")
    print(f"Testing {len(test_questions)} questions with 2 variants\n")
    
    for question_data in test_questions:
        question = question_data['question']
        print(f"Testing: {question[:50]}...")
        
        # Variant A: Current system
        response_a = await rag_service.ask(question=question)
        analysis_a = eval_service._analyze_response(question_data, response_a)
        test.add_result("A", analysis_a['quality_score'], analysis_a['hallucination_score'])
        
        # Variant B: With stricter prompt (you'd modify the system prompt)
        # For now, same system (in real test, you'd change the prompt)
        response_b = await rag_service.ask(question=question)
        analysis_b = eval_service._analyze_response(question_data, response_b)
        test.add_result("B", analysis_b['quality_score'], analysis_b['hallucination_score'])
    
    # Analyze results
    results = test.analyze()
    
    print("\n" + "="*70)
    print("📊 A/B TEST RESULTS")
    print("="*70)
    print(f"\nTest: {results['test_name']}")
    print(f"Sample size: {results['sample_size']}")
    print(f"\nQuality Scores:")
    print(f"  Variant A: {results['quality']['variant_a_mean']}")
    print(f"  Variant B: {results['quality']['variant_b_mean']}")
    print(f"  Difference: {results['quality']['difference']}")
    print(f"  P-value: {results['quality']['p_value']}")
    print(f"  Significant: {results['quality']['significant']}")
    print(f"\nRecommendation: {results['recommendation']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_ab_test())