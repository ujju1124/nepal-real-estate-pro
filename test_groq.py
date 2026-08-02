"""
Quick test to verify Groq API key works
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env file!")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")

# Test Groq connection
try:
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key,
    )
    
    print("\n🧪 Testing Groq API...")
    response = llm.invoke("Say 'Hello from Nepal Real Estate!' in one sentence.")
    print(f"✅ Groq Response: {response.content}")
    print("\n🎉 SUCCESS! Your RAG chatbot is ready to use!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key at: https://console.groq.com/keys")
    print("2. Make sure langchain-groq is installed: pip install langchain-groq")
    print("3. Verify .env file has: GROQ_API_KEY=gsk_...")
