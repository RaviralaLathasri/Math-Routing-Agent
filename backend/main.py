from fastapi import FastAPI
from rag_router import router as rag_router
from rag_router_combined import router as combined_router  # ✅ ADD THIS LINE

app = FastAPI()

app.include_router(rag_router)
app.include_router(combined_router)  # ✅ ADD THIS LINE

@app.get("/")
def read_root():
    return {"message": "Math Routing Agent backend is running"}
