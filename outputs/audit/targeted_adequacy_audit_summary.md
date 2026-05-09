# Targeted Adequacy Audit Summary

This targeted audit supports qualitative example selection and data-quality discussion. Ratings remain small-sample manual judgments; `?` means the example needs verification or the source/reference is too unclear for confident scoring.

## Selection

- adequacy_risk: 6
- baseline_error: 2
- data_quality: 1
- literary_failure: 1
- style_difference: 2
- success: 7
- unclear_fragment: 1

## Gold/Data Quality Review

Gold status:
- gold_needs_edit: 15
- ok: 1
- source_issue: 3
- unclear: 1

Audit decision:
- data_quality_caveat: 2
- exclude_from_adequacy: 4
- keep: 14

## Manual Rating Counts

- teacher: adequacy 1=17, 0=3, ?=0; register 1=20, 0=0, ?=0
- full: adequacy 1=17, 0=3, ?=0; register 1=18, 0=1, ?=1
- lora: adequacy 1=9, 0=10, ?=1; register 1=17, 0=1, ?=2
- adapter: adequacy 1=13, 0=7, ?=0; register 1=1, 0=17, ?=2
- baseline: adequacy 1=10, 0=9, ?=1; register 1=0, 0=19, ?=1

## Recommended Paper Use

- Prefer examples with `audit_decision=keep` for adequacy claims.
- Use `data_quality_caveat` examples only to discuss IruMozhi English/reference noise.
- Do not use `exclude_from_adequacy` examples as evidence for translation adequacy.

## Files

- CSV: `outputs/audit/targeted_adequacy_audit_draft.csv`
- Generator: `data/create_targeted_adequacy_audit.py`
