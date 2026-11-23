"""Generate fixed, heading-aware, and hierarchical chunks for Ablation A1."""

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DIR = BASE_DIR / "warisai_dataset" / "processed" / "cleaned"
CHUNKS_DIR = BASE_DIR / "warisai_dataset" / "processed" / "chunks"
META_DIR = BASE_DIR / "metadata"

HEADING_PATTERNS = {
    "normative": re.compile(r"(?m)^(?=(?:BUKU\s+[IVX]+|BAB\s+[IVX]+|Bagian\s+\w+|Pasal\s+\d+)\b)"),
    "court_decision": re.compile(
        r"(?im)^(?=(?:DUDUK\s+PERKARA|TENTANG\s+DUDUK\s+PERKARA|PERTIMBANGAN\s+HUKUM|"
        r"TENTANG\s+HUKUMNYA|M\s*E\s*N\s*G\s*A\s*D\s*I\s*L\s*I|AMAR\s+PUTUSAN)\b)"
    ),
    "commentary": re.compile(r"(?m)^(?=(?:[A-Z]\.\s+[A-Z]|[1-9]\.\s+[A-Z]|BAB\s+[IVX]+))"),
}


def load_canonical_docs() -> list[dict]:
    """Load cleaned docs, skipping exact duplicates and low-text scans."""
    reg = {
        r["doc_id"]: r
        for r in csv.DictReader((META_DIR / "source_register.csv").open(encoding="utf-8"))
    }
    dedup = json.loads((META_DIR / "dedup_report.json").read_text(encoding="utf-8"))
    skip_ids = set()
    for group in dedup.get("exact_sha256_duplicates", []):
        skip_ids.update(group[1:])

    docs = []
    for doc_id, meta in sorted(reg.items()):
        if doc_id in skip_ids or meta["extraction_status"] != "ok":
            continue
        rec = json.loads((CLEAN_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        rec["meta"] = meta
        docs.append(rec)
    return docs


def doc_with_page_offsets(doc: dict) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages and track (start_char, end_char, page_num) spans."""
    parts, spans, pos = [], [], 0
    for pg in doc.get("pages", []):
        txt = pg["text"].strip()
        if not txt:
            continue
        start = pos
        parts.append(txt)
        pos += len(txt) + 2
        spans.append((start, pos, pg["page"]))
    return "\n\n".join(parts), spans


def pages_for_span(spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    """Find first and last page overlapping character range [start, end]."""
    hit = [p for s, e, p in spans if e >= start and s <= end]
    return (hit[0], hit[-1]) if hit else (1, 1)


def split_words_window(text: str, max_words: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if words else []
    step = max(1, max_words - overlap)
    out = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words]).strip()
        if chunk:
            out.append(chunk)
        if i + max_words >= len(words):
            break
    return out


def make_meta(doc: dict, chunk_id: str, heading: str, p_start: int, p_end: int, text: str, **extra) -> dict:
    m = doc["meta"]
    rec = {
        "chunk_id": chunk_id,
        "doc_id": doc["doc_id"],
        "document_type": doc["document_type"],
        "court": m.get("court", ""),
        "case_date": m.get("case_date", ""),
        "case_class": m.get("case_class", ""),
        "heading": heading,
        "page_start": p_start,
        "page_end": p_end,
        "word_count": len(text.split()),
        "text": text,
    }
    rec.update(extra)
    return rec


def build_fixed(docs: list[dict], max_words: int = 380, overlap: int = 60) -> list[dict]:
    chunks = []
    for doc in docs:
        full_text, spans = doc_with_page_offsets(doc)
        cursor = 0
        for idx, piece in enumerate(split_words_window(full_text, max_words, overlap), 1):
            pos = full_text.find(piece[:40], cursor)
            if pos == -1:
                pos = cursor
            p_s, p_e = pages_for_span(spans, pos, pos + len(piece))
            cursor = pos
            chunks.append(make_meta(doc, f"{doc['doc_id']}_fix_{idx:04d}", "fixed", p_s, p_e, piece))
    return chunks


def split_by_heading(doc: dict) -> list[tuple[str, str, int, int]]:
    full_text, spans = doc_with_page_offsets(doc)
    pat = HEADING_PATTERNS.get(doc["document_type"], HEADING_PATTERNS["commentary"])
    matches = list(pat.finditer(full_text))
    if not matches:
        p_s, p_e = pages_for_span(spans, 0, len(full_text))
        return [("body", full_text, p_s, p_e)]

    sections = []
    if matches[0].start() > 50:
        intro = full_text[: matches[0].start()].strip()
        p_s, p_e = pages_for_span(spans, 0, matches[0].start())
        sections.append(("header_intro", intro, p_s, p_e))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sec_text = full_text[start:end].strip()
        if not sec_text:
            continue
        heading_line = sec_text.splitlines()[0][:80].strip()
        p_s, p_e = pages_for_span(spans, start, end)
        sections.append((heading_line, sec_text, p_s, p_e))
    return sections


def build_heading(docs: list[dict], max_words: int = 420, overlap: int = 50) -> list[dict]:
    chunks = []
    for doc in docs:
        c_idx = 0
        for heading, sec_text, p_s, p_e in split_by_heading(doc):
            for piece in split_words_window(sec_text, max_words, overlap):
                c_idx += 1
                chunks.append(make_meta(doc, f"{doc['doc_id']}_hdg_{c_idx:04d}", heading, p_s, p_e, piece))
    return chunks


def build_hierarchical(docs: list[dict], parent_max: int = 600, child_max: int = 180, overlap: int = 35) -> list[dict]:
    chunks = []
    for doc in docs:
        p_idx, c_idx = 0, 0
        for heading, sec_text, p_s, p_e in split_by_heading(doc):
            for parent_piece in split_words_window(sec_text, parent_max, 60):
                p_idx += 1
                parent_id = f"{doc['doc_id']}_par_{p_idx:04d}"
                for child_piece in split_words_window(parent_piece, child_max, overlap):
                    c_idx += 1
                    chunks.append(
                        make_meta(
                            doc,
                            f"{doc['doc_id']}_hrc_{c_idx:04d}",
                            heading,
                            p_s,
                            p_e,
                            child_piece,
                            parent_id=parent_id,
                            parent_text=parent_piece,
                        )
                    )
    return chunks


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    avg_w = sum(r["word_count"] for r in rows) / max(1, len(rows))
    print(f"Saved {path.name}: {len(rows)} chunks | avg_words={avg_w:.1f}")


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_canonical_docs()
    print(f"Canonical readable docs loaded: {len(docs)}")
    save_jsonl(CHUNKS_DIR / "chunks_fixed.jsonl", build_fixed(docs))
    save_jsonl(CHUNKS_DIR / "chunks_heading.jsonl", build_heading(docs))
    save_jsonl(CHUNKS_DIR / "chunks_hierarchical.jsonl", build_hierarchical(docs))


if __name__ == "__main__":
    main()
