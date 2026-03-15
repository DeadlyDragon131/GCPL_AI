# GCPL AI Hackathon — Option B: RAG System with Retrieval Benchmarking

## Zero Cost Setup — No paid APIs required
| Component       | Tool                        | Cost  |
|-----------------|-----------------------------|-------|
| LLM generation  | Groq (llama-3.1-8b-instant) | Free  |
| Embedding 1     | BAAI/bge-base-en-v1.5       | Free  |
| Embedding 2     | BAAI/bge-small-en-v1.5      | Free  |
| Vector DB       | ChromaDB (local)            | Free  |
| Reranker        | ms-marco-MiniLM-L6 (local)  | Free  |

---

## Quickstart

### 1. Get a free Groq API key (30 seconds)
1. Go to https://console.groq.com
2. Sign up / log in
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`)

### 2. Install & configure
```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your Groq key:
#   GROQ_API_KEY=gsk_your_key_here
```

### 3. Download the corpus
```bash
python data/download_dataset.py
```

### 4. Run the interactive demo
```bash
python src/rag_pipeline.py
```

### 5. Run the full benchmark (all 8 configurations)
```bash
python evaluation/benchmark.py

# Quick mode (3 queries only, faster):
python evaluation/benchmark.py --quick
```

### 6. Analyse results
Open `notebooks/analysis.ipynb` in Jupyter for charts and visualisations.

---

## Project Structure
```
rag_system/
├── config.py                        # Central config (models, paths, hyperparams)
├── requirements.txt
├── .env.example                     # Copy to .env, add Groq key
├── data/
│   └── download_dataset.py          # Downloads 12 papers + synthetic FMCG doc
├── src/
│   ├── ingestion.py                 # Fixed-size & semantic chunking
│   ├── embeddings.py                # BGE-base & BGE-small (both local)
│   ├── vector_store.py              # ChromaDB — one collection per config
│   ├── retrieval.py                 # Vector search & hybrid BM25+RRF
│   ├── reranker.py                  # Cross-encoder reranking (bonus)
│   ├── query_rewriter.py            # HyDE / multi-query / step-back (bonus)
│   ├── generation.py                # Groq LLM answer generation
│   └── rag_pipeline.py              # End-to-end orchestrator + demo
├── evaluation/
│   ├── test_queries.py              # 12 queries with ground truth
│   ├── metrics.py                   # P@k, R@k, F1, MRR, NDCG, qualitative
│   ├── benchmark.py                 # Runs all 8 configs, saves CSVs
│   └── cost_analysis.py             # Latency percentiles & Groq quota tracking
├── notebooks/
│   └── analysis.ipynb               # Charts: comparisons, heatmaps, difficulty
└── results/                         # Auto-generated benchmark outputs
```

## Comparative Tests
| Test | Variable A | Variable B |
|------|-----------|-----------|
| 1 — Chunking   | Fixed-size (512 tok, 64 overlap) | Semantic (cosine threshold 0.85) |
| 2 — Embedding  | BGE-base-en-v1.5 (768-dim)      | BGE-small-en-v1.5 (384-dim)     |
| 3 — Retrieval  | Pure vector (ChromaDB HNSW)     | Hybrid BM25 + Vector + RRF      |

**Total configurations:** 2 × 2 × 2 = **8 pipelines** evaluated on **12 queries**.

## Bonus Features
- **Cross-encoder reranking** (`src/reranker.py`) — ms-marco-MiniLM-L6
- **Query rewriting** (`src/query_rewriter.py`) — HyDE, multi-query, step-back
- **Latency analysis** (`evaluation/cost_analysis.py`) — percentiles + Groq quota tracking
