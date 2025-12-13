import json
from utils.rag_core import CHUNK_SIZE, OVERLAP_RATIO, TOP_K, initialize_components

# API handler function
def handler(request):
    if request.method != 'GET':
        return json.dumps({"error": "Method not allowed"}), 405

    try:
        index, embedding_client = initialize_components()
    except Exception as e:
        return json.dumps({"error": "Internal server error during component initialization.", "detail": str(e)}), 500


    stats = {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
    return json.dumps(stats), 200, {'Content-Type': 'application/json'}

