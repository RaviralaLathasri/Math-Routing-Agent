from fastapi import APIRouter, Query
from vectorstore import search_from_vectorstore

router = APIRouter()

@router.get("/solve")
def solve_question(question: str = Query(..., description="Math question to solve")):
    try:
        # Step 1: Search vector store using the raw question (embedding is done inside the function)
        results = search_from_vectorstore(question)

        # Step 2: Return the most similar match
        if results:
            return {"solution": results[0]["page_content"]}
        else:
            return {"solution": "Not found in knowledge base. Try web search or MCP."}

    except Exception as e:
        return {"error": str(e)}
