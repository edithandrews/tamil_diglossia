#!/usr/bin/env python3
"""Importable Tamil-script span romanizer for code-mixed Tamil/English text."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata


TAMIL_SPAN_RE = re.compile(r"[\u0B80-\u0BFF]+")
RESIDUAL_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
SPACE_RE = re.compile(r"\s+")
ROMAN_WORD_RE = re.compile(r"[a-z\u00c0-\u024f\u1e00-\u1eff]+", re.IGNORECASE)
RomanizedPiece = tuple[str, bool]
RESIDUAL_TAMIL_FALLBACKS = {
    "\u0B82": "m",  # Tamil sign anusvara; Aksharamukha may leave malformed spans unchanged.
}

DIACRITIC_FOLD_MAP = {
    "ā": "aa",
    "ī": "ee",
    "ū": "oo",
    "ē": "e",
    "ō": "o",
    "ḻ": "zh",
    "ḷ": "l",
    "ṟ": "r",
    "ṉ": "n",
    "ṇ": "n",
    "ñ": "nj",
    "ṅ": "ng",
    "ṭ": "t",
    "ḍ": "d",
    "ś": "s",
    "ṣ": "s",
    "ṃ": "m",
    "ṁ": "m",
    "ḥ": "h",
    "ṛ": "r",
}
DIACRITIC_FOLDS = str.maketrans(DIACRITIC_FOLD_MAP)
DIACRITIC_CHARS = frozenset(DIACRITIC_FOLD_MAP)

PRE_FOLD_RULES = (
    (re.compile(r"ñc"), "nj"),
    (re.compile(r"ṅka"), "nga"),
    (re.compile(r"ṉṉu"), "nu"),
    (re.compile(r"(?<!c)cc(?!c)"), "ch"),
    (re.compile(r"(?<!c)c(?![ch])"), "s"),
    (re.compile(r"([aāiīuūeēoō])k(?=[aāiīuūeēoō])"), r"\1g"),
    (re.compile(r"(?<!ṭ)tt(?=[aāiīuūeēoō])"), "th"),
    (re.compile(r"ṭṭ"), "tt"),
    (re.compile(r"(?<!ṭ)ṭ(?!ṭ)(?=[aiuēeoōā])"), "d"),
    (re.compile(r"(?<![ṭt])t(?![ṭth])"), "th"),
)


def _aksharamukha_process():
    """Return Aksharamukha's transliteration function, installing if needed."""

    try:
        from aksharamukha import transliterate  # type: ignore
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "aksharamukha"]
        )
        from aksharamukha import transliterate  # type: ignore
    return transliterate.process


def has_tamil(text: str) -> bool:
    """Return True if text contains Tamil-script characters."""

    return bool(TAMIL_SPAN_RE.search(text))


def aksharamukha_sentence(text: str) -> str:
    """Return text with Tamil-script spans transliterated to ISO 15919.

    Applies only the Aksharamukha Tamil-to-ISO step: no diacritic folding,
    no boundary consonant cleanup, no project-specific rewrite rules. Latin
    spans are passed through unchanged. Useful for preserving the canonical
    ISO romanization alongside the project's plain-ASCII variant.
    """

    if not text or not has_tamil(text):
        return SPACE_RE.sub(" ", text).strip()

    process = _aksharamukha_process()
    return SPACE_RE.sub(
        " ",
        TAMIL_SPAN_RE.sub(lambda m: process("Tamil", "ISO", m.group(0)), text),
    ).strip()


def clean_sentence(text: str) -> str:
    """Return text with only Tamil-script spans romanized and cleaned."""

    if not text or not has_tamil(text):
        return SPACE_RE.sub(" ", text).strip()

    process = _aksharamukha_process()
    pieces: list[RomanizedPiece] = []
    last_end = 0
    for match in TAMIL_SPAN_RE.finditer(text):
        if match.start() > last_end:
            pieces.append((text[last_end : match.start()], False))

        romanized = process("Tamil", "ISO", match.group(0))
        if follows_hyphenated_latin_span(pieces):
            romanized = clean_hyphenated_tamil_suffix(romanized)
        pieces.append((clean_romanized_tamil(romanized), True))
        last_end = match.end()

    if last_end < len(text):
        pieces.append((text[last_end:], False))

    pieces = drop_repeated_tamil_boundary_consonants(pieces)
    joined = "".join(piece for piece, _ in pieces)
    joined = RESIDUAL_TAMIL_RE.sub(
        lambda match: RESIDUAL_TAMIL_FALLBACKS.get(match.group(0), ""),
        joined,
    )
    return SPACE_RE.sub(" ", joined).strip()


def clean_hyphenated_tamil_suffix(romanized: str) -> str:
    """Adjust Tamil suffixes attached to a Latin word by hyphen."""

    if romanized == "m":
        return "um"
    return romanized


def follows_hyphenated_latin_span(pieces: list[RomanizedPiece]) -> bool:
    """Return True when the next Tamil span is attached after a hyphen."""

    return bool(pieces and not pieces[-1][1] and pieces[-1][0].endswith("-"))


def clean_romanized_tamil(text: str) -> str:
    """Convert Aksharamukha ISO Tamil output to this project's plain spelling."""

    def clean_word(match: re.Match[str]) -> str:
        word = match.group(0)
        if not any(ch in DIACRITIC_CHARS for ch in word):
            return word
        for pattern, replacement in PRE_FOLD_RULES:
            word = pattern.sub(replacement, word)
        return fold_diacritics(word)

    return ROMAN_WORD_RE.sub(clean_word, text)


def fold_diacritics(text: str) -> str:
    """Fold ISO romanization diacritics into plain ASCII-ish spelling."""

    folded = text.translate(DIACRITIC_FOLDS)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def drop_repeated_tamil_boundary_consonants(
    pieces: list[RomanizedPiece],
) -> list[RomanizedPiece]:
    """Drop repeated consonants across adjacent Tamil-span boundaries."""

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
        if (
            end_match
            and start_match
            and end_match.group(1).casefold() == start_match.group(1).casefold()
        ):
            updated[idx] = (piece[:-1], is_tamil)
    return updated


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Romanize Tamil-script spans in code-mixed Tamil/English text."
    )
    parser.add_argument("sentence", nargs="?", help="Sentence to clean.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    args = parse_args(argv or sys.argv[1:])
    text = args.sentence if args.sentence is not None else sys.stdin.read()
    print(clean_sentence(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
