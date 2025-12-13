import json
from utils import rag_core # Import the whole module

# API handler function
def handler(request):
    # This endpoint typically handles POST requests with a query in the body
    if request.method != 'POST':
        # Prompting endpoint should only accept POST requests
        return json.dumps({"error": "Method not allowed. Use POST request."}), 405

    try:
        # Load the request body data (where the user query is located)
        data = json.loads(request.body)
        query = data.get("query")
        
        if not query:
            return json.dumps({"error": "Query parameter missing from request body."}), 400

        # 1. Initialize Clients (Pinecone/OpenAI) - Done inside the rag_core file with lazy imports
        index, embedding_client = rag_core.initialize_components()

        # 2. Run the main RAG function
        result = rag_core.answer_ted_query(query, index, embedding_client)

        # 3. Return the final RAG response
        return json.dumps(result), 200, {'Content-Type': 'application/json'}

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format in request body."}), 400
    
    except Exception as e:
        # Fallback for any other unexpected error (e.g., Pinecone/OpenAI connection issue)
        return json.dumps({"error": "An internal processing error occurred.", "detail": str(e)}), 500