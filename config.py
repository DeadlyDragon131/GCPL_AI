"""
config.py — Central configuration for the RAG system.
All tuneable parameters live here so benchmarking is reproducible.

LLM Backend : Groq (free API — get key at console.groq.com)
Embeddings  : Both local (sentence-transformers) — no paid API needed
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file for GROQ_API_KEY

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # optional, not required

# ── LLM Settings (Groq — free, fast) ─────────────────────────────────────────
# Available free Groq models (as of 2025):
#   llama-3.1-8b-instant   — fastest, great for RAG
#   llama-3.3-70b-versatile — stronger reasoning, still free
#   mixtral-8x7b-32768     — large context window
GENERATION_MODEL = "llama-3.1-8b-instant"
TEMPERATURE      = 0.0    # Deterministic for evaluation
MAX_TOKENS       = 512

# ── Embedding Models (Comparative Test 2) — both FREE & LOCAL ────────────────
# No OpenAI key needed for embeddings either.
EMBEDDING_MODELS = {
    "bge_large": "BAAI/bge-base-en-v1.5",    # 768-dim, stronger, ~440MB
    "bge_small": "BAAI/bge-small-en-v1.5",   # 384-dim, faster,  ~130MB
}

# ── Chunking Strategies (Comparative Test 1) ─────────────────────────────────
CHUNKING_STRATEGIES = {
    "fixed": {
        "chunk_size": 512,       # tokens
        "chunk_overlap": 64,
        "description": "Fixed-size with overlap — simple, predictable boundaries",
    },
    "semantic": {
        "breakpoint_threshold": 0.85,  # cosine similarity threshold
        "min_chunk_size": 150,         # words
        "max_chunk_size": 600,         # words
        "description": "Semantic — splits on topic shifts detected by embeddings",
    },
}

# ── Retrieval Settings (Comparative Test 3) ──────────────────────────────────
RETRIEVAL_STRATEGIES = {
    "vector":  {"description": "Pure dense vector search via ChromaDB"},
    "hybrid":  {
        "description": "BM25 (sparse) + vector (dense), RRF fusion",
        "bm25_weight":    0.4,
        "vector_weight":  0.6,
        "rrf_k":          60,   # Reciprocal Rank Fusion constant
    },
}

TOP_K = 5   # Number of chunks to retrieve per query

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/corpus"
RESULTS_DIR  = "results"
CHROMA_DIR   = "data/chroma_db"
