"""Extract text from raw PDFs in Dataset/ and generate metadata/source_register.csv."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
import pypdf

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"
TEXT_OUT_DIR = BASE_DIR / "warisai_dataset" / "processed" / "text"
META_DIR = BASE_DIR / "metadata"

FOLDER_CONFIG = [
    ("KHI", "normative", "KHI"),
    ("Direktori Putusan", "court_decision", "PA"),
    ("Badilag", "commentary", "ART"),
]


def detect_court_and_year(filename: str, first_page: str) -> tuple[str, str]:
    """Infer court name and year from filename or first page text."""
    year_match = re.search(r"_(20\d{2}|19\d{2})_", filename)
    year = year_match.group(1) if year_match else ""
    fn_lower = filename.lower()
    if "_k_ag_" in fn_lower or "_pk_ag_" in fn_lower:
        court = "Mahkamah Agung RI (Kamar Agama)"
    elif "_pta." in fn_lower or "pengadilan tinggi agama" in first_page.lower():
        court = "Pengadilan Tinggi Agama"
    elif "_ms." in fn_lower or "mahkamah syar" in first_page.lower():
        court = "Mahkamah Syar'iyah"
    elif "_pa." in fn_lower or "pengadilan agama" in first_page.lower():
        court = "Pengadilan Agama"
    else:
        court = ""
    return court, year


def main() -> None:
    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for d in [
        "src/retrieval", "src/generation", "src/agent", "src/eval",
        "warisai_dataset/processed/cleaned", "warisai_dataset/processed/chunks",
        "benchmark", "configs", "runs", "logs",
    ]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    register_rows = []
    sha_map = defaultdict(list)
    text_sig_map = defaultdict(list)
    stats = {"total": 0, "ok": 0, "low_text": 0, "error": 0}

    for subfolder, doc_type, prefix in FOLDER_CONFIG:
        folder_path = DATASET_DIR / subfolder
        if not folder_path.exists():
            continue
        pdfs = sorted(folder_path.glob("*.pdf"))
        for idx, pdf_path in enumerate(pdfs, start=1):
            stats["total"] += 1
            doc_id = f"{prefix}_{idx:04d}" if prefix != "KHI" else f"{prefix}_{idx:03d}"
            actual_type = doc_type
            if subfolder == "KHI" and "panduan" in pdf_path.name.lower():
                actual_type = "guidance"

            raw_bytes = pdf_path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            sha_map[sha256].append(doc_id)

            pages = []
            total_chars = 0
            status = "ok"
            err_msg = ""
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                for p_idx, page in enumerate(reader.pages, start=1):
                    txt = page.extract_text() or ""
                    total_chars += len(txt.strip())
                    pages.append({"page": p_idx, "text": txt})
                if total_chars < 300:
                    status = "low_text_needs_ocr"
                    stats["low_text"] += 1
                else:
                    stats["ok"] += 1
            except Exception as exc:
                status = "error"
                err_msg = str(exc)
                stats["error"] += 1

            first_text = pages[0]["text"] if pages else ""
            norm_sig = hashlib.sha256(
                re.sub(r"\s+", " ", "".join(p["text"] for p in pages[:2])).strip()[:3000].encode("utf-8")
            ).hexdigest()
            text_sig_map[norm_sig].append(doc_id)

            court, year = ("", "")
            case_class = ""
            if actual_type == "court_decision":
                court, year = detect_court_and_year(pdf_path.name, first_text)
                case_class = "Perdata Agama - Waris Islam"
            elif actual_type == "normative":
                case_class = "KHI / Inpres 1/1991"

            record = {
                "doc_id": doc_id,
                "document_type": actual_type,
                "source_folder": subfolder,
                "source_file": pdf_path.name,
                "sha256": sha256,
                "num_pages": len(pages),
                "total_chars": total_chars,
                "extraction_status": status,
                "privacy_status": "PENDING_REVIEW" if actual_type == "court_decision" else "NOT_APPLICABLE",
                "pages": pages,
            }
            (TEXT_OUT_DIR / f"{doc_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            register_rows.append({
                "doc_id": doc_id,
                "document_type": actual_type,
                "title": pdf_path.stem,
                "originating_body": court or subfolder,
                "original_url": "",
                "access_date": today,
                "publication_date": year,
                "case_date": year if actual_type == "court_decision" else "",
                "court": court,
                "case_class": case_class,
                "norm_reference": "KHI Buku II" if actual_type == "normative" else "",
                "filename": pdf_path.name,
                "file_format": "pdf",
                "num_pages": len(pages),
                "total_chars": total_chars,
                "sha256": sha256,
                "license_status": "public_regulation" if actual_type == "normative" else "review_required",
                "pii_review_status": "pending" if actual_type == "court_decision" else "not_applicable",
                "extraction_status": status,
                "notes": err_msg,
            })

    reg_path = META_DIR / "source_register.csv"
    with reg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(register_rows[0].keys()))
        writer.writeheader()
        writer.writerows(register_rows)

    dedup = {
        "exact_sha256_duplicates": [v for v in sha_map.values() if len(v) > 1],
        "near_text_duplicates": [v for v in text_sig_map.values() if len(v) > 1],
    }
    (META_DIR / "dedup_report.json").write_text(
        json.dumps(dedup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Extraction summary:", stats)
    print("Exact SHA-256 duplicates:", len(dedup["exact_sha256_duplicates"]))
    print("Near-text duplicates:", len(dedup["near_text_duplicates"]), dedup["near_text_duplicates"])


if __name__ == "__main__":
    main()
