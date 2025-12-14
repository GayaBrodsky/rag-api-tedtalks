import json
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import rag_core
except ImportError:
    # For Vercel deployment
    import rag_core

def handler(event, context):
    """Vercel serverless function handler for stats"""
    if event['httpMethod'] != 'GET':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Method not allowed"})
        }
    
    try:
        # Get stats from rag_core configuration
        stats = {
            "chunk_size": rag_core.CHUNK_SIZE,
            "overlap_ratio": rag_core.OVERLAP_RATIO,
            "top_k": rag_core.TOP_K
        }
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(stats)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Internal server error: {str(e)}"})
        }

