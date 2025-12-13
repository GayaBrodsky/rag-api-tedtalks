import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

df = pd.read_csv('ted_talks_en.csv')

# filter columns 'talk_id', 'title', 'transcript'
df_filtered = df[['talk_id', 'title', 'transcript']]

# drop rows with missing values in 'transcript'
df_filtered = df_filtered.dropna(subset=['transcript'])

### Chunking ####
# Define RAG Hyperparameters
CHUNK_SIZE = 1024
OVERLAP_RATIO = 0.2
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP_RATIO)
TOP_K = 5

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function = len,
    separators=["\n\n", "\n", " ", ""]
)

# Process and structure data
chunk_list = []
for index, row in df_filtered.iterrows():
    chunks = text_splitter.split_text(row['transcript'])
    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk_list.append({
            'talk_id': row['talk_id'],
            'title': row['title'],
            'chunk_index': i,
            'text': chunk
        })

print(f"Total transcripts: {len(df_filtered)}")
print(f"Total chunks created: {len(chunk_list)}")

# Get credentials 
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LLMOD_API_KEY = os.getenv('LLMOD_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

INDEX_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

LLMOD_BASE_URL = "https://api.llmod.ai"

# Embedding client
embedding_client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)

# Function to get index names robustly (handling method vs. property)
def get_index_names(pc_client):
    """Safely retrieves the list of index names."""
    list_result = pc_client.list_indexes()
    
    # Check if .names is a callable method (old client) or a property (new client)
    names = getattr(list_result, 'names', None)
    
    if callable(names):
        return names() # Execute if it's a method
    elif isinstance(names, list):
        return names # Return if it's already a list
    else:
        # Fallback for unexpected format
        return []

# Actual list
index_names = get_index_names(pc)

if INDEX_NAME not in index_names:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=INDEX_DIMENSION,
        metric='cosine', 
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print("Index created successfully.")
else:
    print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

# Get a handle to the index 
index = pc.Index(INDEX_NAME)
print(f"Index is ready. Total vectors in index: {index.describe_index_stats()['total_vector_count']}")

import time 
# Define Batch Size 
BATCH_SIZE = 100 

print(f"\nStarting vector upload of {len(chunk_list)} chunks in batches of {BATCH_SIZE}...")

# The Main Embedding and Upsert Loop (COSTLY STEP) 
for i in range(0, len(chunk_list), BATCH_SIZE):
    i_end = min(len(chunk_list), i + BATCH_SIZE)
    batch = chunk_list[i:i_end]
    
    texts_to_embed = [item['text'] for item in batch]
    
    try:
        # **API CALL: Sends text to the RPRTHPB model via the OpenAI client**
        response = embedding_client.embeddings.create(
            model="RPRTHPB-text-embedding-3-small", 
            input=texts_to_embed
        )
        # Extracts the 1536-dimensional vectors
        embeddings = [item.embedding for item in response.data]
        
    except Exception as e:
        print(f"\nFATAL ERROR during embedding at index {i}. Stopping. Error: {e}")
        # Stop immediately to save budget if the API fails
        break 
        
    # Prepare vectors for Pinecone upload (UPSERT)
    vectors_to_upsert = []
    for j, item in enumerate(batch):
        # Create a unique ID for each chunk
        vector_id = f"{item['talk_id']}-{item['chunk_index']}" 
        
        # Metadata is CRUCIAL for RAG retrieval
        metadata = {
            'talk_id': item['talk_id'],
            'title': item['title'],
            'text': item['text'] # Store the original chunk text
        }
        
        vectors_to_upsert.append((vector_id, embeddings[j], metadata))
        
    # Upload to Pinecone
    index.upsert(vectors=vectors_to_upsert)
    
    # Print status periodically
    if i % 5000 == 0:
        print(f"--> Uploaded {i_end} / {len(chunk_list)} chunks. Time: {time.strftime('%H:%M:%S')}")
        
print("\n--- ✅ All Data Embedded and Uploaded Successfully! ---")
# Final check of the vector count in Pinecone
print(f"Final vector count in Pinecone: {index.describe_index_stats()['total_vector_count']}")