"""
fix_bge_large_index.py
Builds the ChromaDB index for BGE-large (bge-base-en-v1.5) embedding model.
Run: python fix_bge_large_index.py
"""
import sys
sys.path.insert(0, '.')

import config
from src.embeddings import get_embedder
from src.ingestion import ingest_corpus
from src.vector_store import VectorStore

print("Step 1: Loading BGE-large embedder (768-dim)...")
embedder = get_embedder('bge_large')

print("\nStep 2: Ingesting corpus with FIXED chunking...")
chunks = ingest_corpus(
    corpus_dir=config.DATA_DIR,
    strategy='fixed',
    chunk_size=512,
    chunk_overlap=64,
    verbose=True
)

print(f"\nStep 3: Indexing {len(chunks)} chunks with BGE-large embeddings...")
store = VectorStore.create(
    chunking_strategy='fixed',
    embedder=embedder,
    persist_dir=config.CHROMA_DIR,
    reset=True
)
store.add_chunks(chunks, verbose=True)
print(f"\nDone! BGE-large index built with {store.get_chunk_count()} chunks.")

# Also build semantic + bge_large while we're at it
print("\n" + "="*50)
print("Also building SEMANTIC + BGE-large index...")

def embed_fn(texts):
    return embedder.embed(texts)

chunks_sem = ingest_corpus(
    corpus_dir=config.DATA_DIR,
    strategy='semantic',
    embed_fn=embed_fn,
    breakpoint_threshold=0.85,
    verbose=True
)

store_sem = VectorStore.create(
    chunking_strategy='semantic',
    embedder=embedder,
    persist_dir=config.CHROMA_DIR,
    reset=True
)
store_sem.add_chunks(chunks_sem, verbose=True)
print(f"\nDone! Semantic+BGE-large index built with {store_sem.get_chunk_count()} chunks.")
print("\nAll 4 indexes are now ready:")
print("  fixed  + bge_small  (509 chunks)")
print("  semantic + bge_small (763 chunks)")
print(f"  fixed  + bge_large  ({store.get_chunk_count()} chunks)")
print(f"  semantic + bge_large ({store_sem.get_chunk_count()} chunks)")
print("\nYou can now rerun: python demo_all_features.py")