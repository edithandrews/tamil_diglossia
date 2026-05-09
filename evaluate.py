"""
Apply the IruMozhi classifier to all model outputs and compute
bootstrap 95% confidence intervals on % Spoken Tamil.
Also computes a chrF modernity index on the IruMozhi test subset.

Test set: 100 IruMozhi + 200 Tatoeba = 300 total.
Default test set: 100 held-out IruMozhi sentences.
Modernity index: IruMozhi subset only.

Output: outputs/{experiment}/results.csv, outputs/{experiment}/modernity_scores.jsonl
"""

import argparse
import json
import csv
import re
import numpy as np
from pathlib import Path
from transformers import pipeline
from sacrebleu.metrics import BLEU, CHRF
from data.tamil_romanizer import clean_sentence

CLASSIFIER_PATH = "aryaman/xlm-roberta-base-irumozhi"
SPOKEN_LABEL = "LABEL_0"
OUTPUT_DIR = Path("outputs")
N_BOOTSTRAP = 1000
NON_COMPARISON_RE = re.compile(r"[^\w\s]+")
SPACE_RE = re.compile(r"\s+")

METHODS = ["lora", "qlora", "adapter", "full"]
SIZES = [500, 1000, 2000, 3000]
REGIMES = (
    [f"lora_{s}" for s in SIZES]
    + [f"qlora_{s}" for s in SIZES]
    + ["adapter_3000"]          # IA3: one size (negative reference)
    + [f"full_{s}" for s in SIZES]
    + ["baseline_zeroshot"]
)
IRUMOZHI399_REGIMES = ["lora", "baseline_zeroshot"]
GROUPED_REGIMES = ["full", "lora", "lora_r16", "lora_r8", "qlora", "adapter", "baseline_zeroshot"]

PARAM_COUNTS = {
    **{f"lora_{s}": "~1%" for s in SIZES},
    **{f"qlora_{s}": "~1% (4-bit)" for s in SIZES},
    "adapter_3000": "~0.03%",
    **{f"full_{s}": "100%" for s in SIZES},
    "baseline_zeroshot": "0%",
}
IRUMOZHI399_PARAM_COUNTS = {
    "full": "100%",
    "lora": "~1.5% (r=32)",
    "lora_r16": "~0.8% (r=16)",
    "lora_r8": "~0.4% (r=8)",
    "qlora": "~1.5% (r=32, 4-bit)",
    "adapter": "~0.03%",
    "baseline_zeroshot": "0%",
}
GROUPED_PARAM_COUNTS = {
    "full": "100%",
    "lora": "~1.5% (r=32)",
    "lora_r16": "~0.8% (r=16)",
    "lora_r8": "~0.4% (r=8)",
    "qlora": "~1.5% (r=32, 4-bit)",
    "adapter": "~0.03%",
    "baseline_zeroshot": "0%",
}


def load_test_data(include_tatoeba: bool = True) -> tuple[list[dict], list[dict]]:
    irumozhi = [json.loads(l) for l in open(OUTPUT_DIR / "data" / "irumozhi_test.jsonl")]
    tatoeba_path = OUTPUT_DIR / "data" / "tatoeba_test.jsonl"
    tatoeba = [json.loads(l) for l in open(tatoeba_path)] if include_tatoeba and tatoeba_path.exists() else []
    return irumozhi, tatoeba


def load_outputs_aligned(regime: str, experiment_dir: Path, include_tatoeba: bool = False) -> list[dict]:
    path = experiment_dir / f"student_{regime}" / "outputs.jsonl"
    rows = [json.loads(l) for l in open(path)]
    if not include_tatoeba:
        rows = [r for r in rows if r.get("source", "irumozhi") == "irumozhi"]
    rows.sort(key=lambda r: (r.get("source", "irumozhi"), r["idx"]))
    return rows


def load_teacher_texts(irumozhi_idx: list[int], tatoeba_idx: list[int]) -> list[str]:
    """Combined teacher texts: IruMozhi first, Tatoeba second, sorted by idx within each."""
    iru_lookup = {}
    with open(OUTPUT_DIR / "teacher" / "modern-colloquial.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in set(irumozhi_idx):
                iru_lookup[r["idx"]] = r["ta"]

    tat_lookup = {}
    if tatoeba_idx:
        with open(OUTPUT_DIR / "teacher" / "tatoeba_modern-colloquial.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["idx"] in set(tatoeba_idx):
                    tat_lookup[r["idx"]] = r["ta"]

    return [iru_lookup[i] for i in irumozhi_idx] + [tat_lookup[i] for i in tatoeba_idx]


def load_anchor(mode: str, indices: list[int]) -> list[str]:
    """Load IruMozhi teacher anchor texts (modern or classic) for modernity index."""
    lookup = {}
    with open(OUTPUT_DIR / "teacher" / f"{mode}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            lookup[r["idx"]] = r["ta"]
    return [lookup[i] for i in indices]


def classify(texts: list[str], classifier) -> tuple[np.ndarray, np.ndarray]:
    results = classifier([normalize_for_eval(t) for t in texts], truncation=True, max_length=128)
    labels = np.array([1 if r["label"] == SPOKEN_LABEL else 0 for r in results])
    scores = np.array([r["score"] if r["label"] == SPOKEN_LABEL else 1 - r["score"] for r in results])
    return labels, scores


def normalize_for_eval(text: str) -> str:
    """Romanize Tamil-script spans only, preserving existing Latin/romanized text."""
    text = clean_sentence(text).casefold()
    text = text.replace("-", " ")
    text = NON_COMPARISON_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n)]
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(lower), float(upper)


def modernity_scores(student_texts: list[str], modern_texts: list[str], classic_texts: list[str]) -> np.ndarray:
    """Per-sentence chrF ratio: near 1 = modern-like, near 0 = classic-like."""
    chrf = CHRF()
    scores = []
    for s, m, c in zip(student_texts, modern_texts, classic_texts):
        s = normalize_for_eval(s)
        m = normalize_for_eval(m)
        c = normalize_for_eval(c)
        s_m = chrf.sentence_score(s, [m]).score
        s_c = chrf.sentence_score(s, [c]).score
        denom = s_m + s_c
        scores.append(s_m / denom if denom > 0 else 0.5)
    return np.array(scores)


def reference_scores(student_rows: list[dict], irumozhi_test: list[dict]) -> tuple[float, float, float, float]:
    """Score IruMozhi subset against both human annotator references.

    Predictions may be Tamil script, ISO-style romanized Tamil, or code-mixed output.
    normalize_for_eval uses the project Tamil romanizer for Tamil-script spans and
    folds romanized Tamil before metric computation.
    Both annotators are used as multi-reference — post-curation, colloquial_ref becomes
    the curated gold and annotator_2 remains the secondary reference.
    """
    ref1_lookup = {r["idx"]: normalize_for_eval(r["colloquial_ref"]) for r in irumozhi_test}
    ref2_lookup = {
        r["idx"]: normalize_for_eval(r.get("colloquial_annotator_2") or r["colloquial_ref"])
        for r in irumozhi_test
    }
    iru_rows = sorted(
        [r for r in student_rows if r.get("source", "irumozhi") == "irumozhi"],
        key=lambda r: r["idx"],
    )
    preds = [normalize_for_eval(r["generated"]) for r in iru_rows]
    refs1 = [ref1_lookup[r["idx"]] for r in iru_rows]
    refs2 = [ref2_lookup[r["idx"]] for r in iru_rows]

    chrf = CHRF()
    bleu = BLEU()
    sent_chrf = np.array([chrf.sentence_score(pred, [r1, r2]).score for pred, r1, r2 in zip(preds, refs1, refs2)])
    cmean, clo, chi = bootstrap_ci(sent_chrf)
    bleu_score = bleu.corpus_score(preds, [refs1, refs2]).score
    return cmean, clo, chi, float(bleu_score)


def experiment_config(name: str) -> tuple[Path, list[str], dict[str, str], bool]:
    # mBART experiments live under outputs/mbart/<name>/
    if name == "irumozhi399":
        return OUTPUT_DIR / "mbart" / "irumozhi399", GROUPED_REGIMES, IRUMOZHI399_PARAM_COUNTS, False
    if name.startswith("mixed") and name[5:].isdigit():
        return OUTPUT_DIR / "mbart" / name, GROUPED_REGIMES, GROUPED_PARAM_COUNTS, False

    # Decoder experiments live under outputs/decoder/<name without "decoder_" prefix>/
    # Accept both "decoder_<name>" (legacy) and bare "<name>" forms.
    if name.startswith("decoder_"):
        bare = name[len("decoder_"):]
        return OUTPUT_DIR / "decoder" / bare, ["baseline_zeroshot", "lora"], GROUPED_PARAM_COUNTS, False
    if name.startswith("decoder/"):
        bare = name[len("decoder/"):]
        return OUTPUT_DIR / "decoder" / bare, ["baseline_zeroshot", "lora"], GROUPED_PARAM_COUNTS, False

    if name == "legacy":
        return OUTPUT_DIR, ["full", "lora", "adapter", "baseline_zeroshot"], {
            "full": "100%",
            "lora": "~1%",
            "adapter": "~0.03%",
            "baseline_zeroshot": "0%",
        }, False
    return OUTPUT_DIR, REGIMES, PARAM_COUNTS, True


def evaluate_all(experiment: str = "mixed"):
    import torch
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=CLASSIFIER_PATH, device=device)
    experiment_dir, regimes, param_counts, include_tatoeba = experiment_config(experiment)
    irumozhi_test, tatoeba_test = load_test_data(include_tatoeba=include_tatoeba)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    iru_idx = sorted(ex["idx"] for ex in irumozhi_test)
    tat_idx = sorted(ex["idx"] for ex in tatoeba_test)

    rows = []
    modernity_data = {}  # {regime: (indices, en_texts, mod_scores)}

    # Teacher upper bound
    teacher_texts = load_teacher_texts(iru_idx, tat_idx)
    labels, scores = classify(teacher_texts, classifier)
    mean, lo, hi = bootstrap_ci(labels)
    smean, slo, shi = bootstrap_ci(scores)
    teacher_ref_rows = [
        {"source": "irumozhi", "idx": idx, "generated": text}
        for idx, text in zip(iru_idx, load_anchor("modern-colloquial", iru_idx))
    ]
    trchrf, trchrf_lo, trchrf_hi, trbleu = reference_scores(teacher_ref_rows, irumozhi_test)
    rows.append({"condition": "teacher", "n": len(teacher_texts),
                 "mean": mean, "ci_low": lo, "ci_high": hi,
                 "score_mean": smean, "score_ci_low": slo, "score_ci_high": shi,
                 "params": "N/A", "mod_mean": "", "mod_ci_low": "", "mod_ci_high": "",
                 "ref_chrf": trchrf, "ref_chrf_ci_low": trchrf_lo,
                 "ref_chrf_ci_high": trchrf_hi, "ref_bleu": trbleu})

    # Student conditions — skip missing output files gracefully
    for regime in regimes:
        path = experiment_dir / f"student_{regime}" / "outputs.jsonl"
        if not path.exists():
            print(f"  skipping {regime} — outputs not found")
            continue

        output_rows = load_outputs_aligned(regime, experiment_dir, include_tatoeba=include_tatoeba)
        texts = [r["generated"] for r in output_rows]

        labels, scores = classify(texts, classifier)
        mean, lo, hi = bootstrap_ci(labels)
        smean, slo, shi = bootstrap_ci(scores)

        # Modernity on IruMozhi subset only
        iru_rows = sorted(
            [r for r in output_rows if r.get("source", "irumozhi") == "irumozhi"],
            key=lambda r: r["idx"],
        )
        iru_indices = [r["idx"] for r in iru_rows]
        iru_texts = [r["generated"] for r in iru_rows]
        modern_refs = load_anchor("modern-colloquial", iru_indices)
        classic_refs = load_anchor("classic-colloquial", iru_indices)
        mod = modernity_scores(iru_texts, modern_refs, classic_refs)
        mmean, mlo, mhi = bootstrap_ci(mod)
        modernity_data[regime] = (iru_indices, [r["en"] for r in iru_rows], mod)
        rchrf, rchrf_lo, rchrf_hi, rbleu = reference_scores(output_rows, irumozhi_test)

        rows.append({"condition": regime, "n": len(texts),
                     "mean": mean, "ci_low": lo, "ci_high": hi,
                     "score_mean": smean, "score_ci_low": slo, "score_ci_high": shi,
                     "params": param_counts.get(regime, "?"),
                     "mod_mean": mmean, "mod_ci_low": mlo, "mod_ci_high": mhi,
                     "ref_chrf": rchrf, "ref_chrf_ci_low": rchrf_lo,
                     "ref_chrf_ci_high": rchrf_hi, "ref_bleu": rbleu})

    # Save results CSV
    out_path = experiment_dir / "results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "n", "mean", "ci_low", "ci_high",
            "score_mean", "score_ci_low", "score_ci_high",
            "params", "mod_mean", "mod_ci_low", "mod_ci_high",
            "ref_chrf", "ref_chrf_ci_low", "ref_chrf_ci_high", "ref_bleu",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out_path}")

    # Save per-sentence modernity (IruMozhi subset, all available conditions)
    if modernity_data:
        first_regime = next(iter(modernity_data))
        base_indices, base_en, _ = modernity_data[first_regime]
        mod_path = experiment_dir / "modernity_scores.jsonl"
        with open(mod_path, "w") as f:
            for i, (idx, en) in enumerate(zip(base_indices, base_en)):
                row = {"idx": idx, "en": en}
                for regime, (_, _, mod) in modernity_data.items():
                    row[regime] = round(float(mod[i]), 4)
                json.dump(row, f)
                f.write("\n")
        print(f"Per-sentence modernity saved to {mod_path}")

    # Print summary
    print(f"\n{'condition':30s}  {'n':>3}  {'%colloq':>7}  {'95% CI':>13}  {'modernity':>9}  {'95% CI':>13}  {'chrF-ref':>8}  {'BLEU-ref':>8}")
    for row in rows:
        mod_str = (f"{row['mod_mean']:.3f}      [{row['mod_ci_low']:.3f}, {row['mod_ci_high']:.3f}]"
                   if row["mod_mean"] != "" else "       N/A")
        ref_chrf = f"{row['ref_chrf']:.2f}" if row["ref_chrf"] != "" else "N/A"
        ref_bleu = f"{row['ref_bleu']:.2f}" if row["ref_bleu"] != "" else "N/A"
        print(f"{row['condition']:30s}  {row['n']:>3}  {row['mean']:.3f}    "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  {mod_str}  {ref_chrf:>8}  {ref_bleu:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=[
            "mixed", "mixed1000", "mixed2000", "mixed3000", "legacy",
            "irumozhi399", "decoder_irumozhi399",
            "decoder_irumozhi399_qwen05", "decoder_irumozhi399_qwen30",
            "decoder_mixed3000", "decoder_mixed3000_qwen05",
            "decoder_mixed3000_qwen15", "decoder_mixed3000_bloomz",
            "decoder_mixed3000_unsloth_qwen05",
            "decoder_mixed3000_unsloth_qwen15",
            "decoder_mixed3000_unsloth_qwen30",
        ],
        default="mixed3000",
    )
    args = parser.parse_args()
    evaluate_all(args.experiment)
