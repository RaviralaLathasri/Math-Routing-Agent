# build_kb.py

from embedder import get_embedding
from vectorstore import add_to_vectorstore

# 🧠 Math knowledge base (add more if needed)
documents = [
    "The derivative of a function gives the rate at which the function value changes.",
    "Integration is the reverse process of differentiation.",
    "Matrix multiplication is not commutative. A × B ≠ B × A in general.",
    "Probability measures the likelihood of events occurring.",
    "A limit defines the value that a function approaches as the input approaches some value."
]

# 🔍 Step 1: Generate embeddings
embeddings = [get_embedding(doc) for doc in documents]

# 💾 Step 2: Save to FAISS vector store
add_to_vectorstore(embeddings, documents)
