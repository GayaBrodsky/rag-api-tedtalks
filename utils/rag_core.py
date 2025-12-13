import os


# Configuration
CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP_RATIO)
TOP_K = 5
SYSTEM_PROMPT = """
You are a TED Talk assistant that answers questions strictly and only based on the 
TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not 
explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: 
"I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the 
relevant transcript or metadata when helpful.
For multi-result listing queries (e.g., 'list 3 titles'),
prioritize meeting the requested number (e.g., 3) by using any talk title present in the retrieved context,
even if its relevance is partial.
"""

# Retrival Function: embeds the query and retrieves the top_k most relevant chunks
def retrieve_context(query, index, embedding_client, top_k=TOP_K):
    # Embed the query using the RPRTHPB-text-embedding-3-small model
    embedding_response = embedding_client.embeddings.create(
        model="RPRTHPB-text-embedding-3-small",
        input=[query]
    )
    query_vector = embedding_response.data[0].embedding

    # Query Pincecone: search for the most similar chunks
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Format the results
    context_chunks = []
    for match in results['matches']:
        context_chunks.append({
            'talk_id': match.metadata['talk_id'],
            'title': match.metadata['title'],
            'chunk': match.metadata['text'],
            'score': match.score
        })
    return context_chunks

# Prompt augmentation function: formats retrieved context into a single string for the LLM.   
def create_augmented_prompt(query, context_chunks): 
    context_text = "\n\n--- Retrieved Context Chunks ---\n"
    for i, chunk in enumerate(context_chunks):
        context_text += f"SOURCE TALK {i+1} (Title: {chunk['title']}, Talk ID: {chunk['talk_id']}):\n"
        context_text += chunk['chunk'] + "\n---\n"

    user_prompt = f"User question: {query}\n\n{context_text}"
    return user_prompt    

# Generation: calls the RPRTHPB-gpt-5-mini model to generate the final answer
def generate_response(system_prompt, user_prompt, embedding_client):
    response = embedding_client.chat.completions.create(
        model = "RPRTHPB-gpt-5-mini",
        messages = [
            {"role": "system", "content": system_prompt}, # Define the assistant's behavior
            {"role": "user", "content": user_prompt} # Provide the user's question and context
        ],
        temperature = 1.0 
    )
    return response.choices[0].message.content

# Main function to handle the RAG process
def answer_ted_query(query, index, embedding_client):
    from pinecone import Pinecone, ServerlessSpec
    from openai import OpenAI

    # Get credentials (moved from initialize_components)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    LLMOD_API_KEY = os.getenv('LLMOD_API_KEY')
    INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    LLMOD_BASE_URL = "https://api.llmod.ai"

    # Initialize Pinecone and OpenAI Embedding Client (moved from initialize_components)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embedding_client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
    index = pc.Index(INDEX_NAME)
    
    # Retrieve relevant context chunks
    context_chunks = retrieve_context(query, index, embedding_client)

    # Create augmented prompt
    augmented_user_prompt = create_augmented_prompt(query, context_chunks)

    # Generate response using the LLM
    final_response = generate_response(SYSTEM_PROMPT, augmented_user_prompt, embedding_client)

    return {
        "response": final_response,
        "context": context_chunks,
        "Augmented Prompt": {
            "System": SYSTEM_PROMPT,
            "User": augmented_user_prompt
        }               
        }

