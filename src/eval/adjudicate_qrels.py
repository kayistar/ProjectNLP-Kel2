"""Adjudicate blind annotations from Annotator A and B, compute Cohen's kappa, and output qrels.tsv."""

import csv
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

BASE_DIR = Path(__file__).resolve().parents[2]
BENCH_DIR = BASE_DIR / "benchmark"


def load_labels(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["query_id"], r["chunk_id"])] = r
    return rows


def main() -> None:
    p_a = BENCH_DIR / "annotation_annotator_A.csv"
    p_b = BENCH_DIR / "annotation_annotator_B.csv"
    a_rows = load_labels(p_a)
    b_rows = load_labels(p_b)

    y_a, y_b, disagreements, qrels = [], [], [], []
    for key, ra in a_rows.items():
        rb = b_rows.get(key)
        if not rb:
            continue
        la = ra["relevance_label"].strip()
        lb = rb["relevance_label"].strip()
        if la == "" or lb == "":
            continue
        ia, ib = int(la), int(lb)
        y_a.append(ia)
        y_b.append(ib)
        if ia != ib:
            disagreements.append({
                "query_id": key[0],
                "chunk_id": key[1],
                "doc_id": ra["doc_id"],
                "query": ra["query"],
                "label_A": ia,
                "label_B": ib,
                "adjudicated_label": max(ia, ib),  # placeholder until reviewer confirms
                "resolution_notes": "NEEDS_ADJUDICATION",
            })
        final_label = ia if ia == ib else max(ia, ib)
        if final_label > 0:
            qrels.append([key[0], key[1], ra["doc_id"], final_label])

    if not y_a:
        print("Belum ada label yang diisi di annotation_annotator_A.csv dan B.csv.")
        return

    kappa = cohen_kappa_score(y_a, y_b)
    agree_pct = sum(1 for x, y in zip(y_a, y_b) if x == y) / len(y_a) * 100
    print(f"Annotated pairs: {len(y_a)} | Exact agreement: {agree_pct:.1f}% | Cohen's kappa: {kappa:.3f}")
    print(f"Disagreements: {len(disagreements)}")

    with (BENCH_DIR / "disagreement_log.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["query_id", "chunk_id", "doc_id", "query", "label_A", "label_B", "adjudicated_label", "resolution_notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(disagreements)

    with (BENCH_DIR / "qrels.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "chunk_id", "doc_id", "relevance"])
        w.writerows(qrels)


if __name__ == "__main__":
    main()
