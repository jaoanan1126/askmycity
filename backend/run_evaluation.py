"""
Standalone evaluation script
Run this to evaluate your RAG system
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api.rag_service import RAGService
from evaluation import EvaluationService

async def main():
    """Run evaluation"""
    print("🧪 Initializing RAG system...")
    rag_service = RAGService()
    
    print("📋 Loading test cases...")
    eval_service = EvaluationService()
    
    print(f"✓ Loaded {len(eval_service.test_cases)} test questions\n")
    
    # Run evaluation
    results = await eval_service.evaluate_system(
        rag_service=rag_service,
        use_ragas=False,  # Set to True if you want RAGAS metrics
        save_results=True
    )
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {eval_service.results_file}")

if __name__ == "__main__":
    asyncio.run(main())