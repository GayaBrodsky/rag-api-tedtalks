import json
import sys
import os
import traceback

sys.stderr.write = sys.stdout.write

try:
    # Try importing from the same directory (api folder)
    import rag_core
    print("DEBUG: Imported rag_core from api folder")
except ImportError as import_err:
    print(f"DEBUG: First import failed: {import_err}")
    # Try another method if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import rag_core
        print("DEBUG: Imported rag_core after adding path")
    except ImportError:
        print("DEBUG: All import attempts failed")
        # Create empty rag_core to prevent crashes
        rag_core = None


def handler(event, context):
    """Vercel serverless function handler"""
    # Check if rag_core was imported successfully
    if rag_core is None:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Failed to import rag_core module"})
        }
    
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