"""
Day 1: Your first Claude API calls
Learning: How to interact with LLMs via API
"""
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def simple_call():
    """Most basic API call"""
    print("="*60)
    print("1. SIMPLE CALL")
    print("="*60 + "\n")
    
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Explain what a Large Language Model is in 3 sentences."}
        ]
    )
    
    print(message.content[0].text)
    print(f"\nTokens used: {message.usage.input_tokens} in, {message.usage.output_tokens} out")

def with_system_prompt():
    """Using system prompts to control behavior"""
    print("\n" + "="*60)
    print("2. WITH SYSTEM PROMPT")
    print("="*60 + "\n")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI tutor. Explain concepts simply with examples. Use analogies.",
        messages=[
            {"role": "user", "content": "What is RAG in AI?"}
        ]
    )
    
    print(message.content[0].text)

def temperature_experiment():
    """Understanding temperature parameter"""
    print("\n" + "="*60)
    print("3. TEMPERATURE EXPERIMENT")
    print("="*60 + "\n")
    
    question = "Write a creative opening line for a story about AI."
    
    for temp in [0, 0.5, 1.0]:
        print(f"\n--- Temperature: {temp} ---")
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=temp,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        print(message.content[0].text)

def different_models():
    """Comparing different Claude models"""
    print("\n" + "="*60)
    print("4. MODEL COMPARISON")
    print("="*60 + "\n")
    
    question = "Explain transformers in AI in 2 sentences."
    models = [
        "claude-3-5-sonnet-20241022",  # Best balance
        "claude-3-haiku-20240307",     # Fastest, cheapest
    ]
    
    for model in models:
        print(f"\n--- {model} ---")
        
        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        print(message.content[0].text)
        print(f"Tokens: {message.usage.input_tokens} in, {message.usage.output_tokens} out")

if __name__ == "__main__":
    print("🚀 Day 1: First Claude API Calls\n")
    
    simple_call()
    input("\nPress Enter to continue...")
    
    with_system_prompt()
    input("\nPress Enter to continue...")
    
    temperature_experiment()
    input("\nPress Enter to continue...")
    
    different_models()
    
    print("\n✅ Day 1 Complete!")
    print("You now know: API calls, system prompts, temperature, model selection")