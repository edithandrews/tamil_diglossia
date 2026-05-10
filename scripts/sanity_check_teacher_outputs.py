"""
Issue #3 — Classifier early-warning check on teacher outputs.

Runs the IruMozhi classifier over:
  - IruMozhi human references (annotator 1, annotator 2) — colloquial upper bound
  - IruMozhi transliterated column — romanized literary lower bound
  - All generated teacher JSONL files (formal, classic-colloquial, modern-colloquial)

Reports % classified as colloquial (LABEL_0) per source.

Usage (Colab):
    !python scripts/classify_teacher_outputs.py
"""

import json
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline

CLASSIFIER = "aryaman/xlm-roberta-base-irumozhi"
OUTPUT_DIR = Path("outputs")
MODES = ["formal", "classic-colloquial", "modern-colloquial"]


def classify_and_report(clf, texts, label):
    n_colloquial = 0
    results = []
    print(f"\n{label}")
    for result in clf(texts, truncation=True, batch_size=32):
        results.append(result)
        if result["label"] == "LABEL_0":
            n_colloquial += 1
        done = len(results)
        print(f"  {done}/{len(texts)}  —  {n_colloquial/done*100:.1f}% colloquial so far", end="\r")
    pct = n_colloquial / len(results) * 100
    print(f"  FINAL: {pct:.1f}% colloquial  ({n_colloquial}/{len(results)})")
    return pct


if __name__ == "__main__":
    clf = pipeline("text-classification", model=CLASSIFIER, device=0, batch_size=32)
    rows = []

    print("=== IruMozhi human references ===")
    ds = load_dataset("aryaman/irumozhi", split="train")
    for label, texts in [
        ("human ref — annotator 1",          list(ds["colloquial: annotator 1"])),
        ("human ref — annotator 2",          list(ds["colloquial: annotator 2"])),
        ("transliterated literary (lower bound)", list(ds["transliterated"])),
    ]:
        pct = classify_and_report(clf, texts, label)
        rows.append({"source": label, "pct_colloquial": round(pct, 1), "n_total": len(texts)})

    print("\n=== Teacher outputs ===")
    for mode in MODES:
        path = OUTPUT_DIR / f"teacher_{mode}.jsonl"
        if not path.exists():
            print(f"\n{mode}: not found, skipping")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f]
        texts = [r["ta"] for r in records]
        pct = classify_and_report(clf, texts, f"teacher {mode}")
        rows.append({"source": f"teacher {mode}", "pct_colloquial": round(pct, 1), "n_total": len(texts)})

    out_path = OUTPUT_DIR / "classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
