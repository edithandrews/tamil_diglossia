"""
Create a targeted adequacy/register audit sheet for the final write-up.

It samples 20 held-out IruMozhi examples from the mixed3000 experiment and
pre-fills draft adequacy/register judgments for human review. Three of the
selected rows appear in Table 4 of the paper.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


OUTPUT_DIR = Path("outputs")
AUDIT_DIR = OUTPUT_DIR / "audit"
EXPERIMENT_DIR = OUTPUT_DIR / "mbart" / "mixed3000"


AUDIT_ROWS = [
    {
        "idx": 4,
        "bucket": "adequacy_risk",
        "reason": "LoRA is code-mixed but word order is weak; full is a clean command.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "0", "lora_register": "?",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Good example for register not guaranteeing adequacy.",
    },
    {
        "idx": 6,
        "bucket": "style_difference",
        "reason": "Full chooses Tamil-dominant colloquial; LoRA matches English borrowing.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Strong slide example for modernity/style split.",
    },
    {
        "idx": 7,
        "bucket": "unclear_fragment",
        "reason": "Fragmentary English; several plausible renderings.",
        "teacher_adequacy": "?", "teacher_register": "?",
        "full_adequacy": "?", "full_register": "?",
        "lora_adequacy": "?", "lora_register": "1",
        "adapter_adequacy": "?", "adapter_register": "0",
        "baseline_adequacy": "?", "baseline_register": "0",
        "draft_note": "Use as a data-quality caveat, not a scored adequacy example.",
    },
    {
        "idx": 21,
        "bucket": "success",
        "reason": "LoRA/full adopt colloquial light-verb construction; baselines stay literary.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Best all-around qualitative success example.",
    },
    {
        "idx": 26,
        "bucket": "literary_failure",
        "reason": "Adapter/baseline are formal; LoRA looks less colloquial than full.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "?",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Shows register is gradient, not just right/wrong.",
    },
    {
        "idx": 37,
        "bucket": "success",
        "reason": "Full/LoRA capture colloquial negation; zero-shot mistranslates.",
        "teacher_adequacy": "?", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "?",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Teacher may miss the 'also' nuance; verify before citing.",
    },
    {
        "idx": 45,
        "bucket": "adequacy_risk",
        "reason": "LoRA changes meaning toward an imperative; baseline hallucinates songs.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "0", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Good adequacy-risk example for LoRA.",
    },
    {
        "idx": 50,
        "bucket": "adequacy_risk",
        "reason": "Full appears to use wrong lexical item; zero-shot says egg.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "0", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Strong example that colloquial-looking output can be wrong.",
    },
    {
        "idx": 68,
        "bucket": "success",
        "reason": "Full/LoRA transfer colloquial plural and English noun borrowing.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Simple success example.",
    },
    {
        "idx": 88,
        "bucket": "success",
        "reason": "Full/teacher are strong; LoRA may shift possessive from their to your.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "?", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Verify LoRA pronoun before using as a success.",
    },
    {
        "idx": 95,
        "bucket": "baseline_error",
        "reason": "Baseline mistranslates brown as black; trained models preserve meaning.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "?",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Good baseline adequacy failure.",
    },
    {
        "idx": 120,
        "bucket": "adequacy_risk",
        "reason": "LoRA uses avaṅka for an inanimate center; full is cleaner.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "0", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Shows trained colloquial output can have semantic/pronominal error.",
    },
    {
        "idx": 127,
        "bucket": "baseline_error",
        "reason": "Zero-shot mistranslates Mercury as shoe; trained outputs are adequate.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "?",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Clear semantic baseline failure.",
    },
    {
        "idx": 155,
        "bucket": "success",
        "reason": "Full/LoRA preserve meaning with colloquial explain/try construction.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Good broad success example.",
    },
    {
        "idx": 164,
        "bucket": "adequacy_risk",
        "reason": "Full changes voice/person; baseline hallucinates film songs.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "0", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "0", "adapter_register": "0",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Useful for adequacy limitations across conditions.",
    },
    {
        "idx": 170,
        "bucket": "style_difference",
        "reason": "LoRA makes an active 'they launched' reading; full is passive/adequate.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "?", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Verify whether LoRA active voice is acceptable.",
    },
    {
        "idx": 188,
        "bucket": "success",
        "reason": "Full/LoRA produce clear colloquial/code-mixed earthquake sentence.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Good success, though English source is awkward.",
    },
    {
        "idx": 294,
        "bucket": "adequacy_risk",
        "reason": "Full drops release event; baseline changes date/event.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "0", "full_register": "?",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "0", "baseline_register": "0",
        "draft_note": "Strong adequacy contrast between full and LoRA.",
    },
    {
        "idx": 396,
        "bucket": "success",
        "reason": "All preserve meaning; full/LoRA are more colloquial/code-mixed.",
        "teacher_adequacy": "1", "teacher_register": "1",
        "full_adequacy": "1", "full_register": "1",
        "lora_adequacy": "1", "lora_register": "1",
        "adapter_adequacy": "1", "adapter_register": "0",
        "baseline_adequacy": "1", "baseline_register": "0",
        "draft_note": "Clean contrast of register with adequate semantics.",
    },
    {
        "idx": 478,
        "bucket": "data_quality",
        "reason": "English/source has odd 'plaintiff'; outputs inherit likely source noise.",
        "teacher_adequacy": "?", "teacher_register": "1",
        "full_adequacy": "?", "full_register": "1",
        "lora_adequacy": "?", "lora_register": "?",
        "adapter_adequacy": "?", "adapter_register": "0",
        "baseline_adequacy": "?", "baseline_register": "0",
        "draft_note": "Use only as data-quality example.",
    },
]


def load_jsonl_by_idx(path: Path) -> dict[int, dict]:
    with path.open() as f:
        return {json.loads(line)["idx"]: json.loads(line) for line in f}


def output_text(row: dict, key: str = "generated") -> str:
    return row.get(key, "")


def load_existing_audit(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            int(row["idx"]): row
            for row in csv.DictReader(f)
            if row.get("idx", "").isdigit()
        }


def existing_or_draft(existing: dict, draft: dict, key: str) -> str:
    value = existing.get(key, "")
    return value if value != "" else draft[key]


def existing_or_default(existing: dict, key: str, default: str = "") -> str:
    value = existing.get(key, "")
    return value if value != "" else default


def build_rows() -> list[dict]:
    test = load_jsonl_by_idx(OUTPUT_DIR / "data" / "irumozhi_test.jsonl")
    teacher = load_jsonl_by_idx(OUTPUT_DIR / "teacher" / "modern-colloquial.jsonl")
    full = load_jsonl_by_idx(EXPERIMENT_DIR / "student_full" / "outputs.jsonl")
    lora = load_jsonl_by_idx(EXPERIMENT_DIR / "student_lora" / "outputs.jsonl")
    adapter = load_jsonl_by_idx(EXPERIMENT_DIR / "student_adapter" / "outputs.jsonl")
    baseline = load_jsonl_by_idx(EXPERIMENT_DIR / "student_baseline_zeroshot" / "outputs.jsonl")
    existing_rows = load_existing_audit(AUDIT_DIR / "targeted_adequacy_audit_draft.csv")

    rows = []
    for draft in AUDIT_ROWS:
        idx = draft["idx"]
        base = test[idx]
        existing = existing_rows.get(idx, {})
        rows.append({
            "idx": idx,
            "bucket": draft["bucket"],
            "selection_reason": draft["reason"],
            "en": base["en"],
            "human_colloquial_ref": base.get("colloquial_ref", ""),
            "human_annotator_2": base.get("colloquial_annotator_2", ""),
            "gold_verified": existing_or_default(
                existing,
                "gold_verified",
                base.get("colloquial_ref", ""),
            ),
            "gold_status": existing_or_default(existing, "gold_status", "ok"),
            "audit_decision": existing_or_default(existing, "audit_decision", "keep"),
            "gold_notes": existing_or_default(existing, "gold_notes"),
            "teacher_modern": output_text(teacher[idx], "ta"),
            "teacher_adequacy": existing_or_draft(existing, draft, "teacher_adequacy"),
            "teacher_register": existing_or_draft(existing, draft, "teacher_register"),
            "full": output_text(full[idx]),
            "full_adequacy": existing_or_draft(existing, draft, "full_adequacy"),
            "full_register": existing_or_draft(existing, draft, "full_register"),
            "lora": output_text(lora[idx]),
            "lora_adequacy": existing_or_draft(existing, draft, "lora_adequacy"),
            "lora_register": existing_or_draft(existing, draft, "lora_register"),
            "adapter_ia3": output_text(adapter[idx]),
            "adapter_adequacy": existing_or_draft(existing, draft, "adapter_adequacy"),
            "adapter_register": existing_or_draft(existing, draft, "adapter_register"),
            "baseline_zeroshot": output_text(baseline[idx]),
            "baseline_adequacy": existing_or_draft(existing, draft, "baseline_adequacy"),
            "baseline_register": existing_or_draft(existing, draft, "baseline_register"),
            "draft_note": draft["draft_note"],
            "review_note": existing.get("review_note", ""),
        })
    return rows


def summarize(rows: list[dict]) -> str:
    models = ["teacher", "full", "lora", "adapter", "baseline"]
    lines = [
        "# Targeted Adequacy Audit Summary",
        "",
        "This targeted audit supports qualitative example selection and data-quality discussion. Ratings remain small-sample manual judgments; `?` means the example needs verification or the source/reference is too unclear for confident scoring.",
        "",
        "## Selection",
        "",
    ]
    buckets = Counter(row["bucket"] for row in rows)
    for bucket, count in sorted(buckets.items()):
        lines.append(f"- {bucket}: {count}")
    gold_status = Counter(row.get("gold_status", "") for row in rows)
    audit_decision = Counter(row.get("audit_decision", "") for row in rows)
    lines.extend(["", "## Gold/Data Quality Review", ""])
    lines.append("Gold status:")
    for status, count in sorted(gold_status.items()):
        lines.append(f"- {status or '(blank)'}: {count}")
    lines.extend(["", "Audit decision:"])
    for decision, count in sorted(audit_decision.items()):
        lines.append(f"- {decision or '(blank)'}: {count}")
    lines.extend(["", "## Manual Rating Counts", ""])
    for model in models:
        adequacy = Counter(row[f"{model}_adequacy"] for row in rows)
        register = Counter(row[f"{model}_register"] for row in rows)
        lines.append(
            f"- {model}: adequacy 1={adequacy['1']}, 0={adequacy['0']}, ?={adequacy['?']}; "
            f"register 1={register['1']}, 0={register['0']}, ?={register['?']}"
        )
    lines.extend([
        "",
        "## Recommended Paper Use",
        "",
        "- Prefer examples with `audit_decision=keep` for adequacy claims.",
        "- Use `data_quality_caveat` examples only to discuss IruMozhi English/reference noise.",
        "- Do not use `exclude_from_adequacy` examples as evidence for translation adequacy.",
        "",
        "## Files",
        "",
        "- CSV: `outputs/audit/targeted_adequacy_audit_draft.csv`",
        "- Generator: `scripts/data/create_targeted_adequacy_audit.py`",
    ])
    return "\n".join(lines) + "\n"


def summary_json(rows: list[dict]) -> dict:
    models = ["teacher", "full", "lora", "adapter", "baseline"]
    buckets = Counter(row["bucket"] for row in rows)
    gold_status = Counter(row.get("gold_status", "") for row in rows)
    audit_decision = Counter(row.get("audit_decision", "") for row in rows)
    rating_counts = {}
    for model in models:
        adequacy = Counter(row[f"{model}_adequacy"] for row in rows)
        register = Counter(row[f"{model}_register"] for row in rows)
        rating_counts[model] = {
            "adequacy": {"1": adequacy["1"], "0": adequacy["0"], "?": adequacy["?"]},
            "register": {"1": register["1"], "0": register["0"], "?": register["?"]},
        }
    return {
        "status": "reviewed_for_qualitative_use",
        "n_examples": len(rows),
        "selection_buckets": dict(sorted(buckets.items())),
        "gold_status_counts": dict(sorted(gold_status.items())),
        "audit_decision_counts": dict(sorted(audit_decision.items())),
        "rating_counts": rating_counts,
        "recommended_paper_use": [
            "Prefer examples with audit_decision=keep for adequacy claims.",
            "Use data_quality_caveat examples only to discuss IruMozhi English/reference noise.",
            "Do not use exclude_from_adequacy examples as evidence for translation adequacy.",
        ],
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = AUDIT_DIR / "targeted_adequacy_audit_draft.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = AUDIT_DIR / "targeted_adequacy_audit_summary.md"
    summary_path.write_text(summarize(rows))
    summary_json_path = AUDIT_DIR / "targeted_adequacy_audit_summary.json"
    summary_json_path.write_text(json.dumps(summary_json(rows), indent=2))
    print(f"Wrote {len(rows)} audit rows to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote JSON summary to {summary_json_path}")


if __name__ == "__main__":
    main()
