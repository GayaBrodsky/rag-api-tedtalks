import json
from utils import rag_core 

def handler(request):
    if request.method != 'GET':
        return json.dumps({"error": "Method not allowed"}), 405

    try:
        index, embedding_client = rag_core.initialize_components()
    except Exception as e:
            # Fallback if initialization fails
            return json.dumps({"error": "Failed to initialize Pinecone/OpenAI.", "detail": str(e)}), 500

    stats = {
            "chunk_size": rag_core.CHUNK_SIZE,
            "overlap_ratio": rag_core.OVERLAP_RATIO,
            "top_k": rag_core.TOP_K
        }
    return json.dumps(stats), 200, {'Content-Type': 'application/json'}

