from flask import Flask, request, jsonify
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import traceback

load_dotenv()
# Debug: Check if env vars are loaded
print(f"DEBUG: LLMOD_API_KEY loaded: {bool(os.getenv('LLMOD_API_KEY'))}")
print(f"DEBUG: PINECONE_API_KEY loaded: {bool(os.getenv('PINECONE_API_KEY'))}")

app = Flask(__name__)

# Get credentials from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LLMOD_API_KEY = os.getenv('LLMOD_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
LLMOD_BASE_URL = "https://api.llmod.ai"
os.environ['OPENAI_API_KEY'] = LLMOD_API_KEY


# Configuration
TOP_K = 5
SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."""

@app.route('/api/prompt', methods=['POST'])
def prompt():
    """Handle RAG queries"""
    try:
        data = request.get_json()
        query = data.get("question") or data.get("query", "")
        
        if not query:
            return jsonify({"error": "Missing 'question' parameter"}), 400
        
        # Initialize Pinecone and OpenAI
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embedding_client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
        index = pc.Index(INDEX_NAME)
        
        # Embed the query
        embedding_response = embedding_client.embeddings.create(
            model="RPRTHPB-text-embedding-3-small",
            input=[query]
        )
        query_vector = embedding_response.data[0].embedding
        print(f"DEBUG: LLMOD_API_KEY value (first 10 chars): {LLMOD_API_KEY[:10]}...")
        print(f"DEBUG: LLMOD_API_KEY length: {len(LLMOD_API_KEY) if LLMOD_API_KEY else 0}")
        print(f"DEBUG: OPENAI_API_KEY env var: {os.getenv('OPENAI_API_KEY')}")
        
        # Test if we can create a simple client
        try:
            test_client = OpenAI(api_key="test_key")
            print("DEBUG: OpenAI client can be created")
        except Exception as e:
            print(f"DEBUG: OpenAI test creation error: {e}")
        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True
        )
        
        # Format context
        context_chunks = []
        for match in results['matches']:
            context_chunks.append({
                'talk_id': match.metadata['talk_id'],
                'title': match.metadata['title'],
                'chunk': match.metadata['text'],
                'score': match.score
            })
        
        # Create augmented prompt
        context_text = "\n\n--- Retrieved Context Chunks ---\n"
        for i, chunk in enumerate(context_chunks):
            context_text += f"SOURCE {i+1} (Title: {chunk['title']}):\n{chunk['chunk']}\n---\n"
        
        user_prompt = f"User question: {query}\n\n{context_text}"
        
        # Generate response
        response = embedding_client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0
        )
        
        final_response = response.choices[0].message.content
        
        # Return formatted response
        result = {
            "response": final_response,
            "context": context_chunks,
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
            print(f"DEBUG: FULL ERROR: {str(e)}")
            print(f"DEBUG: TRACEBACK:\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Return RAG configuration"""
    stats_data = {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": TOP_K
    }
    return jsonify(stats_data), 200

# Vercel needs this
if __name__ == '__main__':
    app.run()