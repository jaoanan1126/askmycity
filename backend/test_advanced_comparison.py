"""
Compare Basic vs Advanced RAG
Shows improvement from advanced features
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.rag_service import RAGService
from evaluation import EvaluationService

async def compare_basic_vs_advanced():
    """
    Run same questions through basic and advanced RAG
    Compare quality
    """
    print("🆚 Basic RAG vs Advanced RAG Comparison")
    print("="*70)
    
    # Initialize
    print("\n📦 Initializing services...")
    rag_service = RAGService()
    eval_service = EvaluationService()
    
    # Test questions
    test_questions = eval_service.test_cases[:5]
    print(f"✓ Testing with {len(test_questions)} questions\n")
    
    basic_results = []
    advanced_results = []
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case['question']
        print(f"\n[{i}/{len(test_questions)}] {question[:60]}...")
        
        # Basic RAG
        print("  → Basic RAG...", end=" ")
        basic_response = await rag_service.ask(
            question=question,
            use_advanced=False
        )
        basic_analysis = eval_service._analyze_response(test_case, basic_response)
        basic_results.append(basic_analysis)
        print(f"Score: {basic_analysis['quality_score']:.3f}")
        
        # Advanced RAG
        print("  → Advanced RAG...", end=" ")
        advanced_response = await rag_service.ask(
            question=question,
            use_advanced=True
        )
        advanced_analysis = eval_service._analyze_response(test_case, advanced_response)
        advanced_results.append(advanced_analysis)
        print(f"Score: {advanced_analysis['quality_score']:.3f}")
        
        # Show improvement
        improvement = advanced_analysis['quality_score'] - basic_analysis['quality_score']
        if improvement > 0:
            print(f"  ✅ Improvement: +{improvement:.3f}")
        elif improvement < 0:
            print(f"  ⚠️  Worse: {improvement:.3f}")
        else:
            print(f"  ➖ Same")
    
    # Calculate averages
    import statistics
    
    basic_avg = statistics.mean(r['quality_score'] for r in basic_results)
    advanced_avg = statistics.mean(r['quality_score'] for r in advanced_results)
    
    basic_hallucination = statistics.mean(r['hallucination_score'] for r in basic_results)
    advanced_hallucination = statistics.mean(r['hallucination_score'] for r in advanced_results)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n📈 Quality Scores:")
    print(f"  Basic RAG:    {basic_avg:.3f}")
    print(f"  Advanced RAG: {advanced_avg:.3f}")
    print(f"  Improvement:  {'+' if advanced_avg > basic_avg else ''}{advanced_avg - basic_avg:.3f} ({((advanced_avg/basic_avg - 1) * 100):.1f}%)")
    
    print(f"\n🎯 Hallucination Scores:")
    print(f"  Basic RAG:    {basic_hallucination:.3f}")
    print(f"  Advanced RAG: {advanced_hallucination:.3f}")
    print(f"  Reduction:    {basic_hallucination - advanced_hallucination:.3f}")
    
    print(f"\n💡 Interpretation:")
    if advanced_avg > basic_avg + 0.05:
        print("  🎉 Advanced RAG shows significant improvement!")
    elif advanced_avg > basic_avg:
        print("  ✅ Advanced RAG is slightly better")
    else:
        print("  ⚠️  No significant improvement - may need tuning")
    
    print("="*70)

if __name__ == "__main__":
    asyncio.run(compare_basic_vs_advanced())