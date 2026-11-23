"""Evaluate retrieval ablations A1 (chunking) and A2 (BM25 vs Dense vs Hybrid RRF) on qrels."""

import csv
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
BENCH_DIR = BASE_DIR / "benchmark"
INDEX_ROOT = BASE_DIR / "warisai_dataset" / "indexes"
RUNS_DIR = BASE_DIR / "runs"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rel = int(r["relevance"])
            if rel > 0:
                qrels[r["query_id"]][r["doc_id"]] = max(qrels[r["query_id"]].get(r["doc_id"], 0), rel)
    return qrels


def load_queries(split_file: Path) -> list[dict]:
    return [json.loads(line) for line in split_file.open(encoding="utf-8")]


def eval_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    r5, r10, mrr10, ndcg10 = [], [], [], []
    for qid, rel_docs in qrels.items():
        if qid not in rankings:
            continue
        # Deduplicate ranked doc_ids while preserving order
        seen, ranked_docs = set(), []
        for did in rankings[qid]:
            if did not in seen:
                seen.add(did)
                ranked_docs.append(did)

        hits5 = sum(1 for d in ranked_docs[:5] if d in rel_docs)
        hits10 = sum(1 for d in ranked_docs[:10] if d in rel_docs)
        r5.append(hits5 / len(rel_docs))
        r10.append(hits10 / len(rel_docs))

        rr = 0.0
        for rank, d in enumerate(ranked_docs[:10], 1):
            if d in rel_docs:
                rr = 1.0 / rank
                break
        mrr10.append(rr)

        dcg = sum((1.0 / math.log2(rank + 1)) for rank, d in enumerate(ranked_docs[:10], 1) if d in rel_docs)
        ideal_hits = min(len(rel_docs), 10)
        idcg = sum((1.0 / math.log2(rank + 1)) for rank in range(1, ideal_hits + 1))
        ndcg10.append(dcg / idcg if idcg > 0 else 0.0)

    n = max(1, len(r5))
    return {
        "n_queries": len(r5),
        "Recall@5": round(sum(r5) / n, 4),
        "Recall@10": round(sum(r10) / n, 4),
        "MRR@10": round(sum(mrr10) / n, 4),
        "nDCG@10": round(sum(ndcg10) / n, 4),
    }


def retrieve_for_strategy(strategy: str, queries: list[dict], model: SentenceTransformer, top_k: int = 30):
    idx_dir = INDEX_ROOT / strategy
    chunks = [json.loads(line) for line in (idx_dir / "chunks_meta.jsonl").open(encoding="utf-8")]
    with (idx_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    faiss_idx = faiss.read_index(str(idx_dir / "dense.faiss"))

    q_texts = ["query: " + q["query"] for q in queries]
    q_embs = np.asarray(model.encode(q_texts, normalize_embeddings=True), dtype="float32")
    _, dense_ids = faiss_idx.search(q_embs, top_k)

    bm25_res, dense_res, hybrid_res = {}, {}, {}
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        bm_scores = bm25.get_scores(tokenize(q["query"]))
        bm_top = np.argsort(bm_scores)[::-1][:top_k]
        dn_top = dense_ids[qi]

        bm25_res[qid] = [chunks[i]["doc_id"] for i in bm_top]
        dense_res[qid] = [chunks[i]["doc_id"] for i in dn_top if i >= 0]

        rrf = defaultdict(float)
        for rank, ci in enumerate(bm_top, 1):
            rrf[int(ci)] += 1.0 / (60 + rank)
        for rank, ci in enumerate(dn_top, 1):
            if ci >= 0:
                rrf[int(ci)] += 1.0 / (60 + rank)
        hyb_top = sorted(rrf.keys(), key=lambda c: rrf[c], reverse=True)[:top_k]
        hybrid_res[qid] = [chunks[i]["doc_id"] for i in hyb_top]

    return bm25_res, dense_res, hybrid_res


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    qrels_path = BENCH_DIR / "qrels.tsv"
    if not qrels_path.exists():
        qrels_path = BENCH_DIR / "qrels_draft.tsv"
    qrels = load_qrels(qrels_path)
    dev_queries = [q for q in load_queries(BENCH_DIR / "dev.jsonl") if q["answerability"] == "answerable"]

    model = SentenceTransformer("intfloat/multilingual-e5-small", local_files_only=True)
    report = {"qrels_used": qrels_path.name, "results": {}}
    for strat in ["fixed", "heading", "hierarchical"]:
        bm, dn, hy = retrieve_for_strategy(strat, dev_queries, model)
        report["results"][strat] = {
            "bm25": eval_rankings(bm, qrels),
            "dense": eval_rankings(dn, qrels),
            "hybrid_rrf": eval_rankings(hy, qrels),
        }
        print(f"\n=== Strategy: {strat} ===")
        for mode, m in report["results"][strat].items():
            print(f"  {mode:12s}: {m}")

    (RUNS_DIR / "retrieval_ablation_a1_a2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved -> runs/retrieval_ablation_a1_a2.json")


if __name__ == "__main__":
    main()
