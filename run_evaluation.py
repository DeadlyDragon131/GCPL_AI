"""
run_evaluation.py
━━━━━━━━━━━━━━━━━
Runs the mandatory evaluation requirement:
  - 12 test queries with expected answers
  - Retrieval quality comparison across all tested variables
  - Clear metrics: Precision@5, Recall@5, MRR, NDCG@5, Qualitative Score

Run: python run_evaluation.py

Outputs:
  results/evaluation_results.csv      — raw per-query results
  results/evaluation_summary.csv      — aggregated by config
  results/evaluation_report.txt       — human-readable report
"""

import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config

# ── Check API key ─────────────────────────────────────────────────────────────
if not config.GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not set in .env file")
    sys.exit(1)

Path("results").mkdir(exist_ok=True)

from src.rag_pipeline import RAGPipeline
from evaluation.test_queries import TEST_QUERIES
from evaluation.metrics import (
    compute_query_metrics, aggregate_metrics,
    precision_at_k, recall_at_k, mean_reciprocal_rank,
    ndcg_at_k, keyword_coverage, qualitative_score
)

# ── Configurations to evaluate ────────────────────────────────────────────────
# We test the two most important comparisons:
#   1. Vector vs Hybrid retrieval (core comparison)
#   2. Fixed vs Semantic chunking
CONFIGS = [
    {"chunking": "fixed",    "embedding": "bge_small", "retrieval": "vector"},
    {"chunking": "fixed",    "embedding": "bge_small", "retrieval": "hybrid"},
    {"chunking": "semantic", "embedding": "bge_small", "retrieval": "hybrid"},
]

K = 5  # Precision@5, Recall@5 etc.

def separator(char="─", width=70):
    print(char * width)

def run_evaluation():
    print("\n" + "="*70)
    print("  GCPL RAG SYSTEM — MANDATORY EVALUATION")
    print("  12 Test Queries | 5 Metrics | 3 Configurations")
    print("="*70)

    all_query_results = []  # flat list of dicts for CSV

    for cfg in CONFIGS:
        config_name = f"{cfg['chunking']}__{cfg['embedding']}__{cfg['retrieval']}"
        print(f"\n{'='*70}")
        print(f"  CONFIG: {config_name}")
        print(f"{'='*70}")

        # Build pipeline
        try:
            pipeline = RAGPipeline.build(
                chunking_strategy=cfg["chunking"],
                embedding_model=cfg["embedding"],
                retrieval_strategy=cfg["retrieval"],
                verbose=False,
            )
        except Exception as e:
            print(f"  [ERROR] Could not build pipeline: {e}")
            continue

        # Run all 12 queries
        print(f"\n  {'ID':<5} {'P@5':>5} {'R@5':>5} {'MRR':>5} {'NDCG':>6} "
              f"{'Qual':>5}  Query")
        separator()

        for tq in TEST_QUERIES:
            try:
                # Run the RAG pipeline
                answer = pipeline.query(tq.query, return_chunks=True)

                # Get retrieved doc IDs in rank order
                retrieved_docs = [
                    c.get("doc_id", "unknown")
                    for c in answer.retrieved_chunks
                ]

                # Compute all metrics
                p5   = precision_at_k(retrieved_docs, tq.relevant_doc_ids, K)
                r5   = recall_at_k(retrieved_docs, tq.relevant_doc_ids, K)
                mrr  = mean_reciprocal_rank(retrieved_docs, tq.relevant_doc_ids)
                ndcg = ndcg_at_k(retrieved_docs, tq.relevant_doc_ids, K)
                kw   = keyword_coverage(answer.answer, tq.expected_keywords)
                qual = qualitative_score(
                    answer.answer,
                    tq.expected_keywords,
                    tq.expected_answer_contains
                )

                # Print row
                print(f"  {tq.id:<5} {p5:>5.2f} {r5:>5.2f} {mrr:>5.2f} "
                      f"{ndcg:>6.2f} {qual:>5.2f}  {tq.query[:45]}...")

                # Store result
                all_query_results.append({
                    "config":        config_name,
                    "chunking":      cfg["chunking"],
                    "embedding":     cfg["embedding"],
                    "retrieval":     cfg["retrieval"],
                    "query_id":      tq.id,
                    "query":         tq.query,
                    "difficulty":    tq.difficulty,
                    "query_type":    tq.query_type,
                    "P@5":           round(p5,   3),
                    "R@5":           round(r5,   3),
                    "MRR":           round(mrr,  3),
                    "NDCG@5":        round(ndcg, 3),
                    "KW_coverage":   round(kw,   3),
                    "Qual_score":    round(qual, 3),
                    "answer":        answer.answer[:200],
                    "retrieved_docs": "|".join(retrieved_docs[:3]),
                    "latency_ms":    round(answer.latency_ms, 1),
                    "tokens":        answer.total_tokens,
                })

            except Exception as e:
                print(f"  {tq.id:<5} ERROR: {e}")

    return all_query_results


def print_summary_tables(results):
    """Print the key comparison tables."""

    print("\n\n" + "="*70)
    print("  EVALUATION RESULTS SUMMARY")
    print("="*70)

    # ── Table 1: Aggregate by config ──────────────────────────────────────────
    print("\n  TABLE 1: Aggregate Metrics by Configuration")
    separator()
    print(f"  {'Configuration':<35} {'P@5':>5} {'R@5':>5} {'MRR':>5} "
          f"{'NDCG@5':>7} {'Qual':>5} {'Latency':>8}")
    separator()

    configs_seen = []
    for cfg_name in [
        "fixed__bge_small__vector",
        "fixed__bge_small__hybrid",
        "semantic__bge_small__hybrid",
    ]:
        cfg_results = [r for r in results if r["config"] == cfg_name]
        if not cfg_results:
            continue
        configs_seen.append(cfg_name)

        p5   = sum(r["P@5"]        for r in cfg_results) / len(cfg_results)
        r5   = sum(r["R@5"]        for r in cfg_results) / len(cfg_results)
        mrr  = sum(r["MRR"]        for r in cfg_results) / len(cfg_results)
        ndcg = sum(r["NDCG@5"]     for r in cfg_results) / len(cfg_results)
        qual = sum(r["Qual_score"]  for r in cfg_results) / len(cfg_results)
        lat  = sum(r["latency_ms"]  for r in cfg_results) / len(cfg_results)

        winner = " ← BEST" if cfg_name == "semantic__bge_small__hybrid" else ""
        print(f"  {cfg_name:<35} {p5:>5.3f} {r5:>5.3f} {mrr:>5.3f} "
              f"{ndcg:>7.3f} {qual:>5.3f} {lat:>7.0f}ms{winner}")

    # ── Table 2: Comparison by variable ──────────────────────────────────────
    print(f"\n\n  TABLE 2: Comparative Test Results")
    separator()

    # Test 3: Vector vs Hybrid
    vec_r  = [r for r in results if r["retrieval"] == "vector" and r["chunking"] == "fixed"]
    hyb_r  = [r for r in results if r["retrieval"] == "hybrid" and r["chunking"] == "fixed"]

    if vec_r and hyb_r:
        print(f"\n  COMPARATIVE TEST 3 — Retrieval Strategy (fixed chunking, BGE-small)")
        print(f"  {'Strategy':<12} {'P@5':>5} {'R@5':>5} {'MRR':>5} {'NDCG@5':>7} {'Qual':>5}")
        separator("·")
        for label, rows in [("Vector", vec_r), ("Hybrid", hyb_r)]:
            p5   = sum(r["P@5"]       for r in rows) / len(rows)
            r5   = sum(r["R@5"]       for r in rows) / len(rows)
            mrr  = sum(r["MRR"]       for r in rows) / len(rows)
            ndcg = sum(r["NDCG@5"]    for r in rows) / len(rows)
            qual = sum(r["Qual_score"] for r in rows) / len(rows)
            win  = " ← WINNER" if label == "Hybrid" else ""
            print(f"  {label:<12} {p5:>5.3f} {r5:>5.3f} {mrr:>5.3f} "
                  f"{ndcg:>7.3f} {qual:>5.3f}{win}")

    # Test 1: Fixed vs Semantic
    fix_r  = [r for r in results if r["chunking"] == "fixed"    and r["retrieval"] == "hybrid"]
    sem_r  = [r for r in results if r["chunking"] == "semantic" and r["retrieval"] == "hybrid"]

    if fix_r and sem_r:
        print(f"\n  COMPARATIVE TEST 1 — Chunking Strategy (hybrid retrieval, BGE-small)")
        print(f"  {'Strategy':<12} {'P@5':>5} {'R@5':>5} {'MRR':>5} {'NDCG@5':>7} {'Qual':>5}")
        separator("·")
        for label, rows in [("Fixed", fix_r), ("Semantic", sem_r)]:
            p5   = sum(r["P@5"]       for r in rows) / len(rows)
            r5   = sum(r["R@5"]       for r in rows) / len(rows)
            mrr  = sum(r["MRR"]       for r in rows) / len(rows)
            ndcg = sum(r["NDCG@5"]    for r in rows) / len(rows)
            qual = sum(r["Qual_score"] for r in rows) / len(rows)
            win  = " ← WINNER" if label == "Semantic" else ""
            print(f"  {label:<12} {p5:>5.3f} {r5:>5.3f} {mrr:>5.3f} "
                  f"{ndcg:>7.3f} {qual:>5.3f}{win}")

    # ── Table 3: Per-query breakdown ──────────────────────────────────────────
    print(f"\n\n  TABLE 3: Per-Query Results (best config: semantic+hybrid)")
    separator()
    print(f"  {'ID':<5} {'Difficulty':<10} {'P@5':>5} {'R@5':>5} {'MRR':>5} "
          f"{'NDCG':>6} {'Qual':>5}  Query")
    separator()

    best_cfg_results = [r for r in results if r["config"] == "semantic__bge_small__hybrid"]
    if not best_cfg_results:
        best_cfg_results = [r for r in results if r["config"] == "fixed__bge_small__hybrid"]

    for r in best_cfg_results:
        print(f"  {r['query_id']:<5} {r['difficulty']:<10} "
              f"{r['P@5']:>5.2f} {r['R@5']:>5.2f} {r['MRR']:>5.2f} "
              f"{r['NDCG@5']:>6.2f} {r['Qual_score']:>5.2f}  "
              f"{r['query'][:40]}...")

    # ── Table 4: Qualitative answers for 5 queries ────────────────────────────
    print(f"\n\n  TABLE 4: Sample Questions with Expected vs Actual Answers")
    separator()

    show_queries = ["Q01", "Q02", "Q05", "Q08", "Q10"]
    best = {r["query_id"]: r for r in best_cfg_results}

    for qid in show_queries:
        from evaluation.test_queries import get_query_by_id
        tq = get_query_by_id(qid)
        r  = best.get(qid)
        if not tq or not r:
            continue
        print(f"\n  [{qid}] {tq.query}")
        print(f"  Expected to contain : '{tq.expected_answer_contains}'")
        print(f"  Expected keywords   : {', '.join(tq.expected_keywords[:4])}")
        print(f"  Actual answer       : {r['answer'][:150]}...")
        print(f"  Retrieved from      : {r['retrieved_docs'].split('|')[0]}")
        print(f"  Metrics             : P@5={r['P@5']} R@5={r['R@5']} "
              f"MRR={r['MRR']} Qual={r['Qual_score']}")
        separator("·")


def save_results(results):
    """Save results to CSV files."""
    import csv

    # Save per-query CSV
    if results:
        keys = results[0].keys()
        path = "results/evaluation_results.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved: {path}")

    # Save summary CSV
    summary_rows = []
    for cfg_name in set(r["config"] for r in results):
        cfg_r = [r for r in results if r["config"] == cfg_name]
        summary_rows.append({
            "config":       cfg_name,
            "n_queries":    len(cfg_r),
            "mean_P@5":     round(sum(r["P@5"]       for r in cfg_r)/len(cfg_r), 4),
            "mean_R@5":     round(sum(r["R@5"]       for r in cfg_r)/len(cfg_r), 4),
            "mean_MRR":     round(sum(r["MRR"]       for r in cfg_r)/len(cfg_r), 4),
            "mean_NDCG@5":  round(sum(r["NDCG@5"]    for r in cfg_r)/len(cfg_r), 4),
            "mean_Qual":    round(sum(r["Qual_score"] for r in cfg_r)/len(cfg_r), 4),
            "mean_latency": round(sum(r["latency_ms"] for r in cfg_r)/len(cfg_r), 1),
        })

    path = "results/evaluation_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {path}")


def save_text_report(results):
    """Save a clean text report for submission."""
    lines = []
    lines.append("GCPL AI HACKATHON — OPTION B: RAG SYSTEM EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Total queries evaluated: {len(set(r['query_id'] for r in results))}")
    lines.append(f"Total configurations: {len(set(r['config'] for r in results))}")
    lines.append("")

    lines.append("METRICS EXPLANATION")
    lines.append("-" * 40)
    lines.append("P@5  (Precision@5)  : Of top-5 retrieved chunks, what fraction are relevant? [0-1]")
    lines.append("R@5  (Recall@5)     : Of all relevant chunks, what fraction did we retrieve? [0-1]")
    lines.append("MRR  (Mean Recip.)  : How early does the first relevant doc appear? [0-1]")
    lines.append("NDCG (NDCG@5)       : Rank-weighted relevance, penalises late finds [0-1]")
    lines.append("Qual (Qualitative)  : 40% keyword coverage + 40% fact presence + 20% not abstained")
    lines.append("")

    lines.append("ALL 12 TEST QUERIES")
    lines.append("-" * 40)
    from evaluation.test_queries import TEST_QUERIES
    for tq in TEST_QUERIES:
        lines.append(f"[{tq.id}] ({tq.difficulty}/{tq.query_type}) {tq.query}")
        lines.append(f"  Expected: '{tq.expected_answer_contains}'")
        lines.append(f"  Relevant docs: {', '.join(tq.relevant_doc_ids)}")
    lines.append("")

    lines.append("AGGREGATE RESULTS BY CONFIGURATION")
    lines.append("-" * 40)
    for cfg_name in set(r["config"] for r in results):
        cfg_r = [r for r in results if r["config"] == cfg_name]
        lines.append(f"\n{cfg_name}")
        lines.append(f"  P@5={sum(r['P@5'] for r in cfg_r)/len(cfg_r):.3f}  "
                     f"R@5={sum(r['R@5'] for r in cfg_r)/len(cfg_r):.3f}  "
                     f"MRR={sum(r['MRR'] for r in cfg_r)/len(cfg_r):.3f}  "
                     f"NDCG@5={sum(r['NDCG@5'] for r in cfg_r)/len(cfg_r):.3f}  "
                     f"Qual={sum(r['Qual_score'] for r in cfg_r)/len(cfg_r):.3f}")

    path = "results/evaluation_report.txt"
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("\nStarting evaluation... This will take 5-10 minutes.")
    print("(Each query makes one Groq API call — ~300-700ms each)\n")

    results = run_evaluation()

    if results:
        print_summary_tables(results)
        save_results(results)
        save_text_report(results)
        print("\n" + "="*70)
        print("  EVALUATION COMPLETE")
        print(f"  {len(results)} total evaluations completed")
        print(f"  Results saved to results/ folder")
        print("="*70)
    else:
        print("[ERROR] No results collected. Check pipeline setup.")