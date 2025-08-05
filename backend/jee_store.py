import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_jee_vectorstore(filename):
    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)  # This is a list of strings or docs
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
