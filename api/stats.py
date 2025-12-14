import json
import sys
import os

sys.stderr.write = sys.stdout.write

print("DEBUG: Starting stats.py handler")

def handler(event, context):
    """Vercel serverless function handler for stats"""
    print("DEBUG: Stats handler called")
    
    if event.get('httpMethod') != 'GET':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Method not allowed. Use GET."})
        }
    
    # Return hardcoded stats (we'll add rag_core later)
    stats = {
        "chunk_size": 1024,
        "overlap_ratio": 0.2,
        "top_k": 5
    }
    
    print(f"DEBUG: Returning stats: {stats}")
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(stats)
    }

# Vercel needs this export
__all__ = ["handler"]