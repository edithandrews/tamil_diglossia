"""
Convert aksharamukha ISO 15919 romanized Tamil to IruMozhi-style informal romanization.
"""

import re
import unicodedata

_MULTI_CHAR = [
    ("ḻ", "zh"), ("Ḻ", "zh"),
    ("ṅk", "ng"),
    ("ṅ", "ng"), ("Ṅ", "ng"),
    ("ñ", "ny"), ("Ñ", "ny"),
    ("ā", "aa"), ("Ā", "aa"),
    ("ī", "ee"), ("Ī", "ee"),
    ("ū", "oo"), ("Ū", "oo"),
    ("ē", "ee"), ("Ē", "ee"),
    ("ō", "oo"), ("Ō", "oo"),
    ("ṭ", "d"),  ("Ṭ", "d"),
    ("ḍ", "d"),  ("Ḍ", "d"),
    ("ṉ", "n"),  ("Ṉ", "n"),
    ("ṟ", "r"),  ("Ṟ", "r"),
    ("ḷ", "l"),  ("Ḷ", "l"),
    ("ṃ", "m"),  ("Ṃ", "m"),
    ("ḥ", "h"),  ("Ḥ", "h"),
    ("ś", "s"),  ("Ś", "s"),
    ("ṣ", "s"),  ("Ṣ", "s"),
    # ("nt", "ndh"),
    # ("t", "dh"), english loan words are affected
]


def normalize_iso15919(text: str) -> str:
    for old, new in _MULTI_CHAR:
        text = text.replace(old, new)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = text.lower()
    text = text.replace("cc", "ch")
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ref(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
