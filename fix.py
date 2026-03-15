"""
fix_semantic_index.py
Run this to build the semantic chunking vector store.
Usage: python fix_semantic_index.py
"""
import sys
sys.path.insert(0, '.')

import config
from src.embeddings import get_embedder
from src.ingestion import ingest_corpus
from src.vector_store import VectorStore

print("Step 1: Loading BGE-small embedder...")
embedder = get_embedder('bge_small')

def embed_fn(texts):
    return embedder.embed(texts)

print("\nStep 2: Ingesting corpus with SEMANTIC chunking...")
print("(This takes 3-5 minutes — embedding every sentence)")
chunks = ingest_corpus(
    corpus_dir=config.DATA_DIR,
    strategy='semantic',
    embed_fn=embed_fn,
    breakpoint_threshold=0.85,
    verbose=True
)

print(f"\nStep 3: Indexing {len(chunks)} semantic chunks into ChromaDB...")
store = VectorStore.create(
    chunking_strategy='semantic',
    embedder=embedder,
    persist_dir=config.CHROMA_DIR,
    reset=True
)
store.add_chunks(chunks, verbose=True)

print(f"\nDone! Semantic index built with {store.get_chunk_count()} chunks.")
print("You can now rerun: python demo_all_features.py")