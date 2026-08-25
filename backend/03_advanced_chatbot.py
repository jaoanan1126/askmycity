"""
Day 3: Advanced chatbot with streaming and error handling
Learning: Production-ready patterns
"""
import anthropic
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def estimate_tokens(messages):
    """Rough token estimation (1 token ≈ 4 characters)"""
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4

def get_response_with_retry(messages, max_retries=3):
    """Get response with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            # Use streaming for better UX
            with client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
            ) as stream:
                full_response = ""
                
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
                
                print()  # New line after streaming
                return full_response
        
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\n⚠️ Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return "Sorry, I'm experiencing high traffic. Please try again later."
        
        except anthropic.APIError as e:
            print(f"\n❌ API Error: {e}")
            return "Sorry, I encountered an error. Please try again."
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return "Sorry, something went wrong."
    
    return "Maximum retries exceeded."

def production_chatbot():
    """Production-ready chatbot with all features"""
    
    conversation_history = []
    message_count = 0
    total_tokens_used = 0
    
    print("🤖 Production Chatbot (with streaming!)")
    print("="*60)
    print("Commands:")
    print("  'quit'  - Exit")
    print("  'clear' - Reset conversation")
    print("  'stats' - Show statistics")
    print("  'save'  - Save conversation")
    print("="*60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            print("Goodbye! 👋")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            message_count = 0
            print("✓ Conversation cleared\n")
            continue
        
        if user_input.lower() == 'stats':
            tokens = estimate_tokens(conversation_history)
            cost = total_tokens_used * 0.003 / 1000
            print(f"\n📊 Statistics:")
            print(f"   Messages: {message_count}")
            print(f"   Est. current tokens: {tokens}")
            print(f"   Total tokens used: {total_tokens_used}")
            print(f"   Estimated cost: ${cost:.4f}\n")
            continue
        
        if user_input.lower() == 'save':
            filename = f"conversation_{int(time.time())}.txt"
            with open(filename, 'w') as f:
                for msg in conversation_history:
                    f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
            print(f"✓ Saved to {filename}\n")
            continue
        
        # Check token usage
        tokens = estimate_tokens(conversation_history)
        if tokens > 8000:
            print("⚠️ Conversation getting long (>8000 tokens).")
            print("   Consider 'clear' to start fresh or conversation may get expensive.\n")
        
        # Add user message
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get streaming response
        print("\nAssistant: ", end="", flush=True)
        assistant_message = get_response_with_retry(conversation_history)
        print()
        
        # Add to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Update stats
        message_count += 1
        total_tokens_used += estimate_tokens([
            conversation_history[-2],  # User message
            conversation_history[-1]   # Assistant message
        ])

def compare_streaming_vs_blocking():
    """Demonstrate difference between streaming and blocking"""
    
    question = "Write a 200-word explanation of how neural networks learn."
    
    print("🔬 Comparing Streaming vs Blocking\n")
    
    # Blocking (traditional)
    print("1. BLOCKING (wait for complete response):")
    print("   [Waiting", end="", flush=True)
    
    start = time.time()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": question}]
    )
    blocking_time = time.time() - start
    
    for _ in range(5):
        print(".", end="", flush=True)
        time.sleep(0.5)
    
    print("]")
    print(f"   Time: {blocking_time:.2f}s")
    print(f"   Response: {response.content[0].text[:100]}...\n")
    
    # Streaming
    print("2. STREAMING (see tokens as they arrive):")
    print("   ", end="", flush=True)
    
    start = time.time()
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    streaming_time = time.time() - start
    
    print(f"\n\n   Time: {streaming_time:.2f}s")
    print(f"   Much better UX! User sees response immediately.\n")

if __name__ == "__main__":
    print("Choose demo:")
    print("1. Production chatbot (recommended)")
    print("2. Streaming vs blocking comparison")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        production_chatbot()
    elif choice == "2":
        compare_streaming_vs_blocking()
        input("\nPress Enter to try production chatbot...")
        production_chatbot()
    else:
        print("Invalid choice")