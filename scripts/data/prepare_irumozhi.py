"""
Split IruMozhi into train/test and write to outputs/.

Teacher outputs were generated on all 499 examples (not just train).
train_student.py must filter the teacher corpus to train-split idx before training
to avoid leaking test examples. Run this script first to produce the split files.

Outputs:
  outputs/data/irumozhi_train.jsonl  — 399 examples (80%), used for student training
  outputs/data/irumozhi_test.jsonl   — 100 examples (20%), held out for evaluation only

Each line includes:
  {"idx": int, "en": str, "tamil_literary": str,
   "tamil_transliterated": str, "colloquial_ref": str,
   "colloquial_annotator_1": str, "colloquial_annotator_2": str}
  - idx: original index in the HuggingFace dataset (used to align with teacher outputs)
  - tamil_literary: Unicode script, literary Tamil
  - tamil_transliterated: IruMozhi literary Tamil transliteration
  - colloquial_ref: default human reference, currently annotator 1
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

SEED = 42
TEST_SIZE = 100
OUTPUT_DIR = Path("outputs")


def main():
    ds = load_dataset("aryaman/irumozhi", split="train")

    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    test_idx = set(indices[:TEST_SIZE])

    train_rows, test_rows = [], []
    for i, example in enumerate(ds):
        row = {
            "idx":                    i,
            "en":                     example["english"],
            "tamil_literary":         example["tamil"],
            "tamil_transliterated":   example["transliterated"],
            "colloquial_ref":         example["colloquial: annotator 1"],
            "colloquial_annotator_1": example["colloquial: annotator 1"],
            "colloquial_annotator_2": example["colloquial: annotator 2"],
        }
        (test_rows if i in test_idx else train_rows).append(row)

    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write(train_rows, data_dir / "irumozhi_train.jsonl")
    _write(test_rows,  data_dir / "irumozhi_test.jsonl")

    print(f"Train: {len(train_rows)} examples → outputs/data/irumozhi_train.jsonl")
    print(f"Test:  {len(test_rows)} examples → outputs/data/irumozhi_test.jsonl")


def _write(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
