import json
from utils.rag_core import CHUNK_SIZE, OVERLAP_RATIO, TOP_K

# Function to get JSON structure 
def get_stats_config(chunk_size, overlap_ratio, top_k):
    return {
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "top_k": top_k
    }


# API handler function
def handler(request):
    if request.method != 'GET':
        return json.dumps({"error": "Method not allowed"}), 405

    stats = get_stats_config(CHUNK_SIZE, OVERLAP_RATIO, TOP_K)

    return json.dumps(stats), 200, {'Content-Type': 'application/json'}

