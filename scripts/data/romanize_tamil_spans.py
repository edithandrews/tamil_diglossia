#!/usr/bin/env python3
"""Tamil-span romanization for code-mixed Tamil/English output.

This module is intentionally standalone. It romanizes only Tamil-script spans
through Aksharamukha, leaves Latin/code-mixed spans as text, then applies a
small set of project-specific spelling rules for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from data.tamil_romanizer import clean_sentence
except ModuleNotFoundError:
    from tamil_romanizer import clean_sentence


TAMIL_SPAN_RE = re.compile(r"[\u0B80-\u0BFF]+")
SPACE_RE = re.compile(r"\s+")
TAMIL_WORD_RE = re.compile(r"[a-z\u00c0-\u024f\u1e00-\u1eff]+", re.IGNORECASE)

DIACRITIC_FOLDS = str.maketrans(
    {
        "ā": "aa",
        "Ā": "aa",
        "ī": "ee",
        "Ī": "ee",
        "ū": "oo",
        "Ū": "oo",
        "ē": "e",
        "Ē": "e",
        "ō": "o",
        "Ō": "o",
        "ḻ": "zh",
        "Ḻ": "zh",
        "ḷ": "l",
        "Ḷ": "l",
        "ṟ": "r",
        "Ṟ": "r",
        "ṉ": "n",
        "Ṉ": "n",
        "ṇ": "n",
        "Ṇ": "n",
        "ñ": "nj",
        "Ñ": "nj",
        "ṅ": "ng",
        "Ṅ": "ng",
        "ṭ": "t",
        "Ṭ": "t",
        "ḍ": "d",
        "Ḍ": "d",
        "ś": "s",
        "Ś": "s",
        "ṣ": "s",
        "Ṣ": "s",
        "ṃ": "m",
        "Ṃ": "m",
        "ṁ": "m",
        "Ṁ": "m",
        "ḥ": "h",
        "Ḥ": "h",
        "ṛ": "r",
        "Ṛ": "r",
    }
)


LOOSE_TAMIL_WORD_REPLACEMENTS = (
    ("ñc", "nj"),
    ("ṅka", "nga"),
    ("ṉṉum", "num"),
    ("ṉṉu", "nu"),
    ("ṭṭ", "tt"),
)

PRE_FOLD_REGEX_REPLACEMENTS = (
    (re.compile(r"(?<!ṭ)ṭ(?!ṭ)(?=[aiuēeoōā])"), "d"),
)

POST_FOLD_REGEX_REPLACEMENTS = (
    (re.compile(r"(?<!c)cc(?!c)"), "ch"),
    (re.compile(r"(?<!c)c(?![ch])"), "s"),
    (re.compile(r"([aeiou])k(?=[aeiou])"), r"\1g"),
    (re.compile(r"(?<!t)t(?!t|h)"), "th"),
    (re.compile(r"^inth(?=a[pbk]?$)"), "indh"),
    (re.compile(r"^([ai])thoda$"), r"\1dhoda"),
    (re.compile(r"ttai?$"), "tha"),
    (re.compile(r"kkaaka$"), "kkaaga"),
    (re.compile(r"kaaka$"), "kaaga"),
    (re.compile(r"ppo+$"), "ppo"),
)


@dataclass(frozen=True)
class ReviewRow:
    idx: Any
    en: str
    original_output: str
    romanized_output: str


class MissingAksharamukhaError(RuntimeError):
    """Raised when Tamil-script romanization is requested without Aksharamukha."""


def has_tamil(text: str) -> bool:
    return bool(TAMIL_SPAN_RE.search(text))


def _aksharamukha_process():
    try:
        from aksharamukha import transliterate  # type: ignore
    except ImportError as exc:
        raise MissingAksharamukhaError(
            "Tamil-script spans require Aksharamukha. Install it with: "
            "python3 -m pip install aksharamukha"
        ) from exc
    return transliterate.process


def romanize_tamil_spans(
    text: str,
    *,
    source: str = "Tamil",
    target: str = "ISO",
    loosen: bool = False,
) -> str:
    """Romanize Tamil-script spans via Aksharamukha and preserve other spans.

    Latin/code-mixed English text is not passed through Aksharamukha. This avoids
    transliterating names or English words that are already in Latin script.
    """

    if not text or not has_tamil(text):
        return text

    process = _aksharamukha_process()
    pieces: list[tuple[str, bool]] = []
    last_end = 0
    for match in TAMIL_SPAN_RE.finditer(text):
        if match.start() > last_end:
            pieces.append((text[last_end : match.start()], False))
        tamil_span = match.group(0)
        romanized = process(source, target, tamil_span)
        if loosen and pieces and not pieces[-1][1] and pieces[-1][0].endswith("-"):
            if romanized == "m":
                romanized = "um"
            elif romanized.startswith("yōṭa"):
                romanized = romanized[1:]
        if loosen:
            romanized = loosen_tamil_romanization(romanized)
        pieces.append((romanized, True))
        last_end = match.end()
    if last_end < len(text):
        pieces.append((text[last_end:], False))

    if loosen:
        pieces = drop_repeated_tamil_boundary_consonants(pieces)
    return "".join(piece for piece, _is_tamil in pieces)


def fold_iso_diacritics(text: str) -> str:
    """Fold ISO-style romanization marks to ASCII-ish Latin text."""

    folded = text.translate(DIACRITIC_FOLDS)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def loosen_tamil_romanization(text: str) -> str:
    """Move romanized Tamil-script spans toward the review spelling style."""

    def loosen_word(match: re.Match[str]) -> str:
        word = match.group(0)
        loosened = word
        for source, target in LOOSE_TAMIL_WORD_REPLACEMENTS:
            loosened = loosened.replace(source, target)
        for pattern, replacement in PRE_FOLD_REGEX_REPLACEMENTS:
            loosened = pattern.sub(replacement, loosened)
        folded = fold_iso_diacritics(loosened)
        for pattern, replacement in POST_FOLD_REGEX_REPLACEMENTS:
            folded = pattern.sub(replacement, folded)
        return folded

    return TAMIL_WORD_RE.sub(loosen_word, text)


def drop_repeated_tamil_boundary_consonants(
    pieces: list[tuple[str, bool]],
) -> list[tuple[str, bool]]:
    updated = list(pieces)
    for idx, (piece, is_tamil) in enumerate(pieces[:-2]):
        separator, separator_is_tamil = pieces[idx + 1]
        next_piece, next_is_tamil = pieces[idx + 2]
        if not is_tamil or separator_is_tamil or not next_is_tamil:
            continue
        if re.search(r"[A-Za-z0-9]", separator):
            continue
        end_match = re.search(r"([bcdfghjklmnpqrstvwxyz])$", piece, re.IGNORECASE)
        start_match = re.match(r"([bcdfghjklmnpqrstvwxyz])", next_piece, re.IGNORECASE)
        if end_match and start_match and end_match.group(1).casefold() == start_match.group(1).casefold():
            updated[idx] = (piece[:-1], is_tamil)
    return updated


def normalize_for_review(text: str) -> str:
    """Normalize for manual review while keeping Latin case and separators."""

    reviewed = clean_sentence(text)
    return SPACE_RE.sub(" ", reviewed).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(row)
    return rows


def first_present(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None:
            return str(value)
    return ""


def build_review_rows(
    teacher_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]] | None,
    *,
    teacher_fields: list[str],
    student_fields: list[str],
    sample_size: int,
    seed: int,
) -> list[ReviewRow]:
    student_by_idx = {row.get("idx"): row for row in student_rows or []}
    candidates = list(teacher_rows)
    if sample_size and sample_size < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, sample_size)
        candidates.sort(key=lambda row: str(row.get("idx", "")))

    review_rows: list[ReviewRow] = []
    for teacher in candidates:
        idx = teacher.get("idx", "")
        student = student_by_idx.get(idx)
        output_text = (
            first_present(student, student_fields)
            if student
            else first_present(teacher, teacher_fields)
        )
        review_rows.append(
            ReviewRow(
                idx=idx,
                en=first_present(teacher, ["en", "english", "source"]),
                original_output=output_text,
                romanized_output=normalize_for_review(output_text),
            )
        )
    return review_rows


def write_csv(path: Path, rows: list[ReviewRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReviewRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_jsonl(path: Path, rows: list[ReviewRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")


def parse_fields(value: str) -> list[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not fields:
        raise argparse.ArgumentTypeError("field list cannot be empty")
    return fields


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Romanize Tamil-script spans through Aksharamukha, preserve Latin spans, "
            "and write normalized teacher/student review samples."
        )
    )
    parser.add_argument("--teacher", type=Path, help="Teacher JSONL path.")
    parser.add_argument("--student", type=Path, help="Optional student JSONL path.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output review file. Suffix selects format: .csv or .jsonl.",
    )
    parser.add_argument(
        "--teacher-fields",
        type=parse_fields,
        default=["ta_raw", "ta", "target", "text"],
        help="Comma-separated teacher text field preference order.",
    )
    parser.add_argument(
        "--student-fields",
        type=parse_fields,
        default=["generated", "romanized", "ta", "text"],
        help="Comma-separated student text field preference order.",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if not args.teacher or not args.out:
            raise SystemExit("--teacher and --out are required")
        if args.out.suffix not in {".csv", ".jsonl"}:
            raise SystemExit("--out must end in .csv or .jsonl")

        teacher_rows = load_jsonl(args.teacher)
        student_rows = load_jsonl(args.student) if args.student else None
        rows = build_review_rows(
            teacher_rows,
            student_rows,
            teacher_fields=args.teacher_fields,
            student_fields=args.student_fields,
            sample_size=args.sample_size,
            seed=args.seed,
        )

        if args.out.suffix == ".csv":
            write_csv(args.out, rows)
        else:
            write_jsonl(args.out, rows)
        print(f"Wrote {len(rows)} rows to {args.out}")
        return 0
    except MissingAksharamukhaError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
