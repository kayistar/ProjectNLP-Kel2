"""Build BM25 and FAISS dense vector indexes for WarisAI."""

import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_DIR = BASE_DIR / "warisai_dataset" / "processed" / "chunks"
INDEX_ROOT = BASE_DIR / "warisai_dataset" / "indexes"

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"


def tokenize_bm25(text: str) -> list[str]:
    """Simple Indonesian-friendly alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(strategy: str) -> list[dict]:
    path = CHUNKS_DIR / f"chunks_{strategy}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build_bm25(chunks: list[dict], out_dir: Path) -> None:
    tokenized = [tokenize_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 saved -> {out_dir / 'bm25.pkl'}")


def build_dense(chunks: list[dict], out_dir: Path, model_name: str, batch_size: int = 64) -> None:
    model = SentenceTransformer(model_name)
    prefix = "passage: " if "e5" in model_name.lower() else ""
    texts = [prefix + c["text"] for c in chunks]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_dir / "dense.faiss"))
    (out_dir / "index_config.json").write_text(
        json.dumps({"embed_model": model_name, "dim": int(emb.shape[1]), "num_chunks": len(chunks)}, indent=2),
        encoding="utf-8",
    )
    print(f"  FAISS saved -> {out_dir / 'dense.faiss'} (dim={emb.shape[1]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["fixed", "heading", "hierarchical"])
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    for strat in args.strategies:
        print(f"Building indexes for strategy={strat}...")
        chunks = load_chunks(strat)
        out_dir = INDEX_ROOT / strat
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "chunks_meta.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        build_bm25(chunks, out_dir)
        if not args.bm25_only:
            build_dense(chunks, out_dir, args.model)


if __name__ == "__main__":
    main()
