# vectorstore.py

import faiss
import os
import pickle
import numpy as np
from embedder import get_embedding

VECTOR_DB_PATH = "vectorstore/faiss_index"
DOC_STORE_PATH = "vectorstore/doc_store.pkl"

# Ensure vectorstore directory exists
os.makedirs("vectorstore", exist_ok=True)

doc_store = []
index = None

def add_to_vectorstore(embeddings, docs):
    """
    Adds embeddings and their corresponding documents to FAISS and saves them.
    """
    global doc_store, index

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and documents
    faiss.write_index(index, VECTOR_DB_PATH)
    doc_store = docs
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(doc_store, f)

    print("✅ Vectorstore built and saved.")


def search_from_vectorstore(query_text, top_k=1):
    """
    Encodes query and searches FAISS for the most similar document.
    """
    global doc_store, index

    # Load FAISS index and docs if not already loaded
    if index is None:
        if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(DOC_STORE_PATH):
            raise ValueError("Vectorstore not found. Please run build_kb.py first.")
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(DOC_STORE_PATH, "rb") as f:
            doc_store = pickle.load(f)

    # Get embedding
    query_embedding = get_embedding(query_text)
    query_embedding = np.array([query_embedding]).astype("float32")

    # Search
    D, I = index.search(query_embedding, k=top_k)

    results = []
    for idx in I[0]:
        if idx < len(doc_store):
            results.append({"page_content": doc_store[idx]})
        else:
            results.append({"page_content": "No relevant document found."})

    return results
