from flask import Flask, request, jsonify
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2
TOP_K = 5

SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data." Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful. For 'list exactly 3' or similar multi-result requests, select the requested number of talks from the retrieved context that are most related to the topic, even if their relevance is not perfect. Prioritize meeting the requested number."""

# Get credentials
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LLMOD_API_KEY = os.getenv('LLMOD_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
LLMOD_BASE_URL = "https://api.llmod.ai"

def initialize_clients():
    """Initialize Pinecone and OpenAI clients"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embedding_client = OpenAI(
        api_key=LLMOD_API_KEY,
        base_url=LLMOD_BASE_URL
    )
    index = pc.Index(INDEX_NAME)
    return index, embedding_client

def retrieve_context(query, index, embedding_client, top_k=TOP_K):
    """Retrieve relevant context chunks from Pinecone"""
    # Embed query
    embedding_response = embedding_client.embeddings.create(
        model="RPRTHPB-text-embedding-3-small",
        input=[query]
    )
    query_vector = embedding_response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    context_chunks = []
    for match in results['matches']:
        context_chunks.append({
            'talk_id': match.metadata['talk_id'],
            'title': match.metadata['title'],
            'chunk': match.metadata['text'],
            'score': match.score
        })
    
    return context_chunks

def create_augmented_prompt(query, context_chunks):
     """Create augmented prompt (for JSON output - just the question)"""
     return f"User question: {query}"

def generate_response(system_prompt, query, context_chunks, embedding_client):
    """Generate response using LLM with context"""
    # Build the actual prompt for the LLM (with context)
    context_text = "\n\n--- Retrieved Context Chunks ---\n"
    for i, chunk in enumerate(context_chunks):
        context_text += f"SOURCE {i+1} (Title: {chunk['title']}):\n{chunk['chunk']}\n---\n"
    
    llm_prompt = f"User question: {query}\n\n{context_text}"
    
    # Call the LLM
    response = embedding_client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_prompt}  # Use llm_prompt WITH context
        ],
        temperature=1.0
    )
    return response.choices[0].message.content


@app.route('/api/prompt', methods=['POST'])
def prompt():
    """Handle RAG queries"""
    try:
        # Get query from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("question") or data.get("query", "")
        if not query:
            return jsonify({"error": "Missing 'question' parameter"}), 400
        
        # Initialize clients
        index, embedding_client = initialize_clients()
        
        # Retrieve context
        context_chunks = retrieve_context(query, index, embedding_client)
        
        # Create augmented prompt
        user_prompt = create_augmented_prompt(query, context_chunks)
        
        # Generate response
        final_response = generate_response(SYSTEM_PROMPT, query, context_chunks, embedding_client)

        
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
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Return RAG configuration"""
    stats_data = {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
    return jsonify(stats_data), 200

if __name__ == '__main__':
    app.run(debug=True)