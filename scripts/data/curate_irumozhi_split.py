"""
Export and apply a manually curated IruMozhi test split.

Workflow:
  1. Export all 499 current IruMozhi rows to a CSV:
       python scripts/data/curate_irumozhi_split.py --export

  2. Open outputs/data/irumozhi_test_candidates.csv and delete rows until exactly
     100 acceptable test sentences remain. Keep the idx column intact.

  3. Apply the curated CSV:
       python scripts/data/curate_irumozhi_split.py --apply outputs/data/irumozhi_test_candidates.csv

This writes:
  outputs/data/irumozhi_test.jsonl   — the 100 curated test rows
  outputs/data/irumozhi_train.jsonl  — the remaining 399 train rows

The teacher files are keyed by idx, so they do not need to be regenerated after
changing the split.
"""

import argparse
import csv
import json
from pathlib import Path

OUTPUT_DIR = Path("outputs")
DATA_DIR = OUTPUT_DIR / "data"
TRAIN_PATH = DATA_DIR / "irumozhi_train.jsonl"
TEST_PATH = DATA_DIR / "irumozhi_test.jsonl"
DEFAULT_CANDIDATE_PATH = DATA_DIR / "irumozhi_test_candidates.csv"
EXPECTED_TEST_SIZE = 100
RAW_ARROW_PATH = Path(
    "/home/edith/.cache/huggingface/datasets/aryaman___irumozhi/default/0.0.0/"
    "fa969ba381a39b120eaab5ed97b012c917c97a00/irumozhi-train.arrow"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_raw_irumozhi_by_idx() -> dict[int, dict]:
    """Load raw IruMozhi columns from the local Arrow cache when available."""
    if RAW_ARROW_PATH.exists():
        import pyarrow.ipc as ipc
        with RAW_ARROW_PATH.open("rb") as f:
            table = ipc.open_stream(f).read_all()
        return {i: row for i, row in enumerate(table.to_pylist())}

    from datasets import load_dataset
    ds = load_dataset("aryaman/irumozhi", split="train")
    return {i: example for i, example in enumerate(ds)}


def load_all_rows() -> list[dict]:
    rows = []
    for split, path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
        for row in load_jsonl(path):
            rows.append({**row, "original_split": split})

    seen = set()
    duplicates = []
    for row in rows:
        idx = row["idx"]
        if idx in seen:
            duplicates.append(idx)
        seen.add(idx)
    if duplicates:
        raise ValueError(f"Duplicate idx values in current split files: {duplicates[:10]}")
    if len(rows) != 499:
        raise ValueError(f"Expected 499 IruMozhi rows, found {len(rows)}")

    raw_by_idx = load_raw_irumozhi_by_idx()
    enriched = []
    for row in rows:
        raw = raw_by_idx[row["idx"]]
        enriched.append({
            **row,
            "tamil_transliterated": row.get("tamil_transliterated", raw["transliterated"]),
            "colloquial_annotator_1": row.get("colloquial_annotator_1", raw["colloquial: annotator 1"]),
            "colloquial_annotator_2": row.get("colloquial_annotator_2", raw["colloquial: annotator 2"]),
        })
    return sorted(enriched, key=lambda r: r["idx"])


def export_candidates(path: Path):
    rows = load_all_rows()
    fieldnames = [
        "idx",
        "original_split",
        "en",
        "colloquial_annotator_1",
        "colloquial_annotator_2",
        "my_gold_colloquial",
        "gold_notes",
        "tamil_literary",
        "tamil_transliterated",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "idx": row["idx"],
                "original_split": row["original_split"],
                "en": row["en"],
                "colloquial_annotator_1": row["colloquial_annotator_1"],
                "colloquial_annotator_2": row["colloquial_annotator_2"],
                "my_gold_colloquial": "",
                "gold_notes": "",
                "tamil_literary": row["tamil_literary"],
                "tamil_transliterated": row["tamil_transliterated"],
                "notes": "",
            })
    print(f"Wrote {len(rows)} candidate rows to {path}")
    print(f"Delete rows until exactly {EXPECTED_TEST_SIZE} test candidates remain, then run --apply.")


def read_curated_rows(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "idx" not in (reader.fieldnames or []):
            raise ValueError("Curated CSV must contain an idx column")
        rows = {int(row["idx"]): row for row in reader if row.get("idx", "").strip()}

    if len(rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Curated CSV must contain exactly {EXPECTED_TEST_SIZE} unique idx values; "
            f"found {len(rows)}"
        )
    return rows


def write_jsonl(rows: list[dict], path: Path):
    keep_fields = [
        "idx",
        "en",
        "tamil_literary",
        "tamil_transliterated",
        "colloquial_ref",
        "colloquial_annotator_1",
        "colloquial_annotator_2",
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {field: row[field] for field in keep_fields}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def apply_curated_split(path: Path):
    rows = load_all_rows()
    curated_rows = read_curated_rows(path)
    test_idx = set(curated_rows)

    for row in rows:
        if row["idx"] in curated_rows:
            curated = curated_rows[row["idx"]]
            gold = curated.get("my_gold_colloquial", "").strip()
            row["colloquial_ref"] = gold or row["colloquial_annotator_1"]

    test_rows = [row for row in rows if row["idx"] in test_idx]
    train_rows = [row for row in rows if row["idx"] not in test_idx]

    if len(test_rows) != EXPECTED_TEST_SIZE or len(train_rows) != 499 - EXPECTED_TEST_SIZE:
        raise ValueError(f"Bad split sizes: train={len(train_rows)}, test={len(test_rows)}")

    write_jsonl(train_rows, TRAIN_PATH)
    write_jsonl(test_rows, TEST_PATH)

    print(f"Wrote train split: {len(train_rows)} rows -> {TRAIN_PATH}")
    print(f"Wrote test split : {len(test_rows)} rows -> {TEST_PATH}")
    print("Teacher files are idx-aligned; no teacher regeneration is needed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help=f"Write candidate CSV to {DEFAULT_CANDIDATE_PATH}")
    parser.add_argument("--apply", type=Path,
                        help="Apply an edited candidate CSV with exactly 100 remaining idx rows")
    args = parser.parse_args()

    if args.export == bool(args.apply):
        parser.error("Choose exactly one of --export or --apply PATH")

    if args.export:
        export_candidates(DEFAULT_CANDIDATE_PATH)
    else:
        apply_curated_split(args.apply)


if __name__ == "__main__":
    main()
