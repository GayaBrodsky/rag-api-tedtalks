# TED Talk RAG API

This project is a Retrieval-Augmented Generation (RAG) system for TED Talk transcripts. It uses Pinecone for vector search and an LLM to answer user questions based on retrieved context.

## Project Structure
- `api/index.py`: The main Flask application and API endpoints.
- `rag_query_test.py`: A script to test the RAG functionality locally.
- `requirements.txt`: Python dependencies.
- `vercel.json`: Deployment configuration for Vercel.

## API Endpoints
- **GET** `/api/stats`: View RAG configuration.
- **POST** `/api/prompt`: Submit a JSON body with `{"query": "your question"}` to get an answer.

## Local Setup
1. Clone the repo: `git clone https://github.com/GayaBrodsky/rag-api-tedtalks.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API keys as environment variables (`PINECONE_API_KEY`, `LLMOD_API_KEY`, `PINECONE_INDEX_NAME`).
4. Run the app: `python api/index.py`
