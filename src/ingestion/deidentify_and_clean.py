"""Clean extracted text and de-identify PII in court decisions."""

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TEXT_IN_DIR = BASE_DIR / "warisai_dataset" / "processed" / "text"
CLEAN_OUT_DIR = BASE_DIR / "warisai_dataset" / "processed" / "cleaned"
META_DIR = BASE_DIR / "metadata"

WATERMARK_PATTERNS = [
    r"(?im)^\s*Mahkamah Agung Republik Indonesia\s*$",
    r"(?im)^\s*Direktori Putusan Mahkamah Agung Republik Indonesia\s*$",
    r"(?im)^\s*putusan\.mahkamahagung\.go\.id\s*$",
]

PII_RULES = [
    ("nik", re.compile(r"\bNIK\s*[:.]?\s*\d{12,18}\b", re.IGNORECASE), "NIK: [NIK_REDACTED]"),
    ("phone", re.compile(r"(?:\+62|62|0)8\d{1,3}[-\s]?\d{3,4}[-\s]?\d{3,5}"), "[PHONE_REDACTED]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),
    (
        "address_line",
        re.compile(
            r"(?i)(bertempat\s+tinggal\s+di|alamat\s+di|beralamat\s+di)\s+[^.;\n]{10,160}"
        ),
        r"\1 [ADDRESS_REDACTED]",
    ),
]


def clean_page_text(text: str) -> str:
    """Remove repetitive MA headers and normalize whitespace."""
    for pat in WATERMARK_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Apply regex PII masking and return counts per category."""
    counts = {}
    for name, pattern, repl in PII_RULES:
        text, n = pattern.subn(repl, text)
        counts[name] = n
    return text, counts


def main() -> None:
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    privacy_rows = []
    for json_path in sorted(TEXT_IN_DIR.glob("*.json")):
        rec = json.loads(json_path.read_text(encoding="utf-8"))
        doc_id = rec["doc_id"]
        doc_type = rec["document_type"]
        total_pii = {"nik": 0, "phone": 0, "email": 0, "address_line": 0}

        cleaned_pages = []
        for pg in rec.get("pages", []):
            txt = clean_page_text(pg["text"])
            if doc_type == "court_decision":
                txt, pg_counts = mask_pii(txt)
                for k, v in pg_counts.items():
                    total_pii[k] += v
            cleaned_pages.append({"page": pg["page"], "text": txt})

        pii_status = "auto_masked_needs_spotcheck" if doc_type == "court_decision" else "not_applicable"
        out_rec = {
            "doc_id": doc_id,
            "document_type": doc_type,
            "source_file": rec["source_file"],
            "sha256": rec["sha256"],
            "num_pages": rec["num_pages"],
            "privacy_status": pii_status,
            "pages": cleaned_pages,
        }
        (CLEAN_OUT_DIR / f"{doc_id}.json").write_text(
            json.dumps(out_rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        privacy_rows.append({
            "doc_id": doc_id,
            "document_type": doc_type,
            "pii_review_status": pii_status,
            "nik_masked": total_pii["nik"],
            "phone_masked": total_pii["phone"],
            "email_masked": total_pii["email"],
            "address_masked": total_pii["address_line"],
            "total_masked": sum(total_pii.values()),
        })

    priv_path = META_DIR / "privacy_review.csv"
    with priv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(privacy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(privacy_rows)

    total_docs = len(privacy_rows)
    total_masked = sum(r["total_masked"] for r in privacy_rows)
    print(f"Cleaned & processed {total_docs} documents. Total PII spans masked: {total_masked}")


if __name__ == "__main__":
    main()
