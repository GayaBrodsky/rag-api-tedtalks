import json
from utils.rag_core import CHUNK_SIZE, OVERLAP_RATIO, TOP_K

# API handler function
def handler(request):
    if request.method != 'GET':
        return json.dumps({"error": "Method not allowed"}), 405

    stats = {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
    return json.dumps(stats), 200, {'Content-Type': 'application/json'}

