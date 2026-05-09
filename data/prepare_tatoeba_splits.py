"""
Split classifier-passed Tatoeba sentences into train and test sets.

Reads:  outputs/data/tatoeba_colloquial.jsonl                     (3166 classifier-passed sentences)
        outputs/teacher/tatoeba_modern-colloquial.jsonl      (teacher translations with romanization)

Writes: outputs/data/tatoeba_test.jsonl   (200 held-out test examples)
        outputs/data/tatoeba_train.jsonl  (2966 training examples, pre-shuffled seed=42)

The train file is ordered so that cumulative subsets (first N rows) are valid
training sets — train_student.py takes the first (size - 399) rows for each run.
"""

import json
import random
from pathlib import Path

SEED = 42
TEST_SIZE = 200
OUTPUT_DIR = Path("outputs")


def main():
    data_dir = OUTPUT_DIR / "data"
    teacher_dir = OUTPUT_DIR / "teacher"
    data_dir.mkdir(parents=True, exist_ok=True)

    col_en = {json.loads(l)["en"] for l in open(data_dir / "tatoeba_colloquial.jsonl")}

    examples = []
    with open(teacher_dir / "tatoeba_modern-colloquial.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["en"] in col_en:
                examples.append(r)

    rng = random.Random(SEED)
    rng.shuffle(examples)

    test_examples = examples[:TEST_SIZE]
    train_examples = examples[TEST_SIZE:]

    with open(data_dir / "tatoeba_test.jsonl", "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(data_dir / "tatoeba_train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Test:  {len(test_examples)} → outputs/data/tatoeba_test.jsonl")
    print(f"Train: {len(train_examples)} → outputs/data/tatoeba_train.jsonl")


if __name__ == "__main__":
    main()
