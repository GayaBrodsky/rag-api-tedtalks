import json
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import rag_core
except ImportError:
    # For Vercel deployment
    import rag_core

def handler(event, context):
    """Vercel serverless function handler"""
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Method not allowed. Use POST."})
        }
    
    try:
        # Parse the request body
        data = json.loads(event.get('body', '{}'))
        query = data.get("query") or data.get("question")  # Support both "query" and "question"
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "Missing 'question' or 'query' parameter"})
            }
        
        # Initialize components
        index, embedding_client = rag_core.initialize_components()
        
        # Run RAG query
        result = rag_core.answer_ted_query(query, index, embedding_client)
        
        # Format response to match assignment requirements
        formatted_result = {
            "response": result.get("response", ""),
            "context": result.get("context", []),
            "Augmented_prompt": {
                "System": result.get("Augmented Prompt", {}).get("System", rag_core.SYSTEM_PROMPT),
                "User": result.get("Augmented Prompt", {}).get("User", "")
            }
        }
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(formatted_result)
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Invalid JSON format"})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Internal server error: {str(e)}"})
        }