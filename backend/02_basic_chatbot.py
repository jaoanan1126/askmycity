"""
Day 2: Chatbot with conversation memory
Learning: How to maintain context across multiple turns
"""
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def simple_chatbot():
    """Basic chatbot with memory"""
    
    conversation_history = []
    
    print("🤖 Simple Chatbot")
    print("Commands: 'quit' to exit, 'clear' to reset, 'stats' for info\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Commands
        if user_input.lower() == 'quit':
            print("Goodbye! 👋")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("✓ Conversation cleared\n")
            continue
        
        if user_input.lower() == 'stats':
            print(f"📊 Conversation length: {len(conversation_history)//2} turns\n")
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=conversation_history
            )
            
            print(response)
            assistant_message = response.content[0].text
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            print(f"\nAssistant: {assistant_message}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def chatbot_with_personality():
    """Chatbot with custom system prompt"""
    
    conversation_history = []
    
    # Different personalities to try
    personalities = {
        "tutor": "You are a friendly AI tutor helping someone learn about generative AI. Use simple explanations and examples.",
        "concise": "You are a concise assistant. Provide brief, direct answers. Use bullet points.",
        "creative": "You are a creative writer. Use vivid language and metaphors.",
        "technical": "You are a senior engineer. Provide detailed technical explanations."
    }
    
    print("🎭 Chatbot with Personality")
    print("Available personalities:", ", ".join(personalities.keys()))
    
    personality = input("Choose personality (or press Enter for default): ").strip().lower()
    system_prompt = personalities.get(personality, None)
    
    if system_prompt:
        print(f"✓ Using '{personality}' personality\n")
    else:
        print("✓ Using default personality\n")
    
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            break
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Build API call
        api_params = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": conversation_history
        }
        
        if system_prompt:
            api_params["system"] = system_prompt
        
        response = client.messages.create(**api_params)
        assistant_message = response.content[0].text
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        print(f"\nAssistant: {assistant_message}\n")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Simple chatbot")
    print("2. Chatbot with personality")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        simple_chatbot()
    elif choice == "2":
        chatbot_with_personality()
    else:
        print("Invalid choice")