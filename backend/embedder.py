from sentence_transformers import SentenceTransformer
import numpy as np

# Load the lightweight model (~100MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    """
    Converts input text to a 384-dimension embedding.
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype("float32")

