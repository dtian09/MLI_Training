import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
import gensim.downloader as api


app = FastAPI()

# Define the model in the global scope
model = api.load('glove-wiki-gigaword-50')

# Define the collection in the global scope using the new client format
client = chromadb.PersistentClient(path="./chroma_db")

## Initialize ChromaDB collection
#collection = client.get_or_create_collection(name="marco_sn_all_documents") # with 600k documents
collection = client.get_or_create_collection(name="marco_sn_documents_avg_pool") # with 150k documents

# Define the avg_pool function before it's used
def avg_pool(sentence):
    wrds = sentence.lower().split()
    embs = [model[w] for w in wrds if w in model]
    if embs: 
        return np.mean(embs, axis=0)
    else: 
        return np.zeros(model.vector_size)

# Define input schema
class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/search")
def search(query: str = Query(..., description="Put in your query here")):
    query_embedding = avg_pool(query)
    results = collection.query(
        query_texts=[query],
        query_embeddings=[query_embedding],  # Wrap in a list as expected by ChromaDB
        n_results=5
    )

    documents = results['documents'][0]
    distances = results['distances'][0]
    clean_results = [{"rank": i+1, "text": doc, "distance": dist} for i, (doc, dist) in enumerate(zip(documents, distances))]

    return clean_results