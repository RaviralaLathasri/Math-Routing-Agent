# build_jee_kb.py

# build_jee_kb.py

from jee_benchmarks import jee_qa
from embedder import get_embedding
from jee_store import save_vectorstore


jee_vectorstore = []

for item in jee_qa:
    embedding = get_embedding(item["question"])
    jee_vectorstore.append({
        "question": item["question"],
        "answer": item["answer"],
        "embedding": embedding
    })

save_vectorstore(jee_vectorstore, "jee_vectorstore.pkl")
print("✅ JEE Vectorstore built and saved as jee_vectorstore.pkl")
