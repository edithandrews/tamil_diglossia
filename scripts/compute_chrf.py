"""
Compute chrF and BLEU for all student conditions in an experiment:
  - vs. human IruMozhi references (normalized)
  - vs. Sarvam teacher modern-colloquial outputs

Uses the pre-normalized `romanized` field from student outputs.
No classifier or GPU needed.

Usage:
    uv run python3 compute_chrf.py --experiment mixed3000
"""

import argparse
import json
import re
import csv
from pathlib import Path
from sacrebleu.metrics import CHRF, BLEU
from data.tamil_romanizer import clean_sentence

OUTPUT_DIR = Path("outputs")


def normalize_ref(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(experiment: str):
    experiment_dir = OUTPUT_DIR / experiment
    test_rows = [json.loads(l) for l in open(OUTPUT_DIR / "irumozhi_test.jsonl")]
    test_idx = {r["idx"] for r in test_rows}
    ref1 = {r["idx"]: normalize_ref(r["colloquial_ref"]) for r in test_rows}
    ref2 = {
        r["idx"]: normalize_ref(r.get("colloquial_annotator_2") or r["colloquial_ref"])
        for r in test_rows
    }

    teacher_lookup = {}
    with open(OUTPUT_DIR / "teacher_modern-colloquial.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in test_idx:
                teacher_lookup[r["idx"]] = normalize_ref(clean_sentence(r["ta_raw"]))

    regimes = sorted(
        p.name.removeprefix("student_")
        for p in experiment_dir.glob("student_*")
        if (p / "outputs.jsonl").exists()
    )

    chrf = CHRF()
    bleu = BLEU()
    results = []

    print(f"Experiment: {experiment}")
    print(f"{'condition':<25}  {'chrF (ref)':>10}  {'BLEU (ref)':>10}  {'chrF (teacher)':>14}  {'BLEU (teacher)':>14}")
    print("-" * 80)

    for regime in regimes:
        path = experiment_dir / f"student_{regime}" / "outputs.jsonl"
        rows = sorted(
            [json.loads(l) for l in open(path) if json.loads(l).get("source", "irumozhi") == "irumozhi"],
            key=lambda r: r["idx"],
        )
        if not rows:
            print(f"  {regime:<23}  (no irumozhi rows)")
            continue

        preds = [r.get("romanized", r["generated"]).lower() for r in rows]
        r1s = [ref1[r["idx"]] for r in rows]
        r2s = [ref2[r["idx"]] for r in rows]
        teacher_refs = [teacher_lookup[r["idx"]] for r in rows]

        chrf_ref = chrf.corpus_score(preds, [r1s, r2s]).score
        bleu_ref = bleu.corpus_score(preds, [r1s, r2s]).score
        chrf_teacher = chrf.corpus_score(preds, [teacher_refs]).score
        bleu_teacher = bleu.corpus_score(preds, [teacher_refs]).score

        results.append({
            "condition": regime,
            "chrF_vs_ref": round(chrf_ref, 2),
            "BLEU_vs_ref": round(bleu_ref, 2),
            "chrF_vs_teacher": round(chrf_teacher, 2),
            "BLEU_vs_teacher": round(bleu_teacher, 2),
        })
        print(f"  {regime:<23}  {chrf_ref:>10.2f}  {bleu_ref:>10.2f}  {chrf_teacher:>14.2f}  {bleu_teacher:>14.2f}")

    out_path = experiment_dir / "chrf_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "chrF_vs_ref", "BLEU_vs_ref", "chrF_vs_teacher", "BLEU_vs_teacher"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="mixed3000")
    args = parser.parse_args()
    main(args.experiment)
