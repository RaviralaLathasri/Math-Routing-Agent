# Math Routing Agent

Math-Routing-Agent is a small project that routes math/JEExam questions to appropriate retrieval-augmented-generation (RAG) components. It includes a backend for embedding and vector store management and a minimal frontend app.

## Repository structure

- `backend/` - core Python components:
  - `main.py` - entry point for backend (RAG routing examples)
  - `embedder.py`, `vectorstore.py` - embedding and vectorstore utilities
  - `build_kb.py`, `build_jee_kb.py`, `generate_jee_vectorstore.py` - knowledge base/vectorstore build scripts
  - other helpers for feedback and benchmarks
- `frontend/` - lightweight front-end app (Flask/Streamlit style entry)

## Requirements

Install Python dependencies listed in `backend/requirements.txt`:

```bash
python -m pip install -r backend/requirements.txt
```

## Running locally

1. Prepare the vectorstore / embeddings using the backend scripts as needed (examples in `backend/`).
2. Start the backend:

```bash
python backend/main.py
```

3. Start the frontend (example):

```bash
python frontend/app.py
```

## Notes

- This README provides a brief overview. Inspect the `backend/` scripts for usage details and configuration options (API keys, model choices, storage paths).
- To push changes to GitHub, ensure your local environment has git credentials configured.

## Remote

This repository will be pushed to: https://github.com/RaviralaLathasri/Math-Routing-Agent.git

---

If you want, I can expand the README with usage examples, environment variables, or deployment instructions.