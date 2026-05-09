"""
Filter Tatoeba en–ta sentence pairs through the IruMozhi classifier.
Keeps only pairs where the Tamil side is classified as colloquial.

Input:  JSONL with {"idx": int, "en": str, "ta": str, "ta_raw": str}
          - ta     = romanized Tamil (ISO 15919) — used for classification
          - ta_raw = Unicode Tamil — kept as training target

Output:
  outputs/data/tatoeba_colloquial.jsonl  — {"en": str, "ta_raw": str}
  No idx field → train_student.py treats these as always-included augmentation.

Resumable: outputs/data/tatoeba_filter_progress.jsonl records every classified
sentence (kept and rejected). Re-running skips already-classified sentences
and appends only new results.

Usage:
  python scripts/data/filter_tatoeba.py --input data/tatoeba_en_ta.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline

CLASSIFIER = "aryaman/xlm-roberta-base-irumozhi"
OUTPUT_DIR = Path("outputs")
DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "tatoeba_colloquial.jsonl"
PROGRESS_PATH = DATA_DIR / "tatoeba_filter_progress.jsonl"
BATCH_SIZE = 32


def load_pairs(input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_seen() -> set[str]:
    """English sentences already classified (kept or rejected)."""
    if not PROGRESS_PATH.exists():
        return set()
    seen = set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen.add(json.loads(line)["en"])
    return seen


def is_colloquial(result: dict) -> bool:
    """
    Decide whether a classifier result is colloquial enough to keep.

    result = {"label": "LABEL_0" | "LABEL_1", "score": float}
    LABEL_0 = Colloquial, LABEL_1 = Literary.

    Trade-off: a stricter score threshold gives a cleaner training signal
    but fewer examples; any LABEL_0 maximises recall.

    """
    return result["label"] == "LABEL_0" and result["score"] >= 0.9


def main(input_path: Path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(input_path)
    seen = load_seen()
    pending = [ex for ex in pairs if ex["en"] not in seen]

    print(f"Total pairs   : {len(pairs)}")
    print(f"Already done  : {len(seen)}")
    print(f"Pending       : {len(pending)}")

    if not pending:
        n = sum(1 for _ in open(OUTPUT_PATH)) if OUTPUT_PATH.exists() else 0
        print(f"Nothing to do. Colloquial output: {n} pairs in {OUTPUT_PATH}")
        return

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=CLASSIFIER, device=device, batch_size=BATCH_SIZE)

    n_colloquial = 0
    n_done = 0

    with open(PROGRESS_PATH, "a", encoding="utf-8") as progress_f, \
         open(OUTPUT_PATH, "a", encoding="utf-8") as output_f:

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]
            results = clf([ex["ta"] for ex in batch], truncation=True, batch_size=BATCH_SIZE)

            for ex, result in zip(batch, results):
                progress_f.write(
                    json.dumps(
                        {"en": ex["en"], "label": result["label"], "score": round(result["score"], 4)},
                        ensure_ascii=False,
                    ) + "\n"
                )
                if is_colloquial(result):
                    n_colloquial += 1
                    output_f.write(
                        json.dumps({"en": ex["en"], "ta_raw": ex["ta_raw"]}, ensure_ascii=False) + "\n"
                    )

            progress_f.flush()
            output_f.flush()
            n_done += len(batch)
            print(
                f"  {n_done}/{len(pending)}  —  {n_colloquial} colloquial "
                f"({n_colloquial / n_done * 100:.1f}%)",
                end="\r",
            )

    print(f"\nDone. {n_colloquial}/{len(pending)} new pairs → {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with en/ta/ta_raw fields")
    args = parser.parse_args()
    main(Path(args.input))
