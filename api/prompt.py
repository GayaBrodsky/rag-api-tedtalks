import json
from utils.rag_core import index, embedding_client, answer_ted_query

# Handles the POST request for /api/prompt
def handler(request):
    if request.method != 'POST':
        return json.dumps({"error": "Method not allowed"}), 405

    try:
        data = json.loads(request.body)
        query = data.get('query')
        if not query:
            return json.dumps({"error": "Missing 'query' field"}), 400

        # Execute the full validated RAG pipeline
        result = answer_ted_query(query, index, embedding_client)

        return json.dumps(result), 200, {'Content-Type': 'application/json'}

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON body"}), 400
    except Exception as e:
        # Catch RAG/LLM errors for debugging
        return json.dumps({"error": f"Internal RAG Error: {str(e)}"}), 500