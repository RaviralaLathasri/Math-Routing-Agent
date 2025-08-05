from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from vectorstore import search_from_vectorstore as search_paper
from jee_store import load_jee_vectorstore
from embedder import get_embedding
import numpy as np

router = APIRouter()

# Load JEE vectorstore (list of documents)
jee_vectorstore = load_jee_vectorstore("jee_vectorstore.pkl")
if not jee_vectorstore or not isinstance(jee_vectorstore, list):
    raise RuntimeError("Failed to load valid JEE vectorstore.")

# Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 1

@router.post("/rag_combined/")
def rag_combined_search(request: QueryRequest):
    query = request.query
    top_k = int(request.top_k)  # Ensure it's a plain int

    try:
        # Embed the query
        query_embedding = np.array([get_embedding(query)], dtype="float32")

        # 🔍 Search paper vectorstore
        paper_results = search_paper(query, top_k=top_k)
        if not paper_results:
            raise HTTPException(status_code=404, detail="No results found in paper vectorstore.")

        paper_text = paper_results[0]['page_content']
        paper_embedding = np.array([get_embedding(paper_text)], dtype="float32")
        paper_score = float(np.linalg.norm(query_embedding - paper_embedding))  # convert to float

        # 🔍 Search JEE vectorstore (brute-force)
        jee_embeddings = np.array([get_embedding(doc) for doc in jee_vectorstore], dtype="float32")
        jee_dists = np.linalg.norm(jee_embeddings - query_embedding, axis=1)
        jee_best_idx = int(np.argmin(jee_dists))  # convert np.int64 to int
        jee_best_doc = jee_vectorstore[jee_best_idx]
        jee_score = float(jee_dists[jee_best_idx])  # convert to float

        # ✅ Compare and return best
        if jee_score < paper_score:
            return {
                "answer": jee_best_doc,
                "source": "jee",
                "score": jee_score
            }
        else:
            return {
                "answer": paper_text,
                "source": "paper",
                "score": paper_score
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
