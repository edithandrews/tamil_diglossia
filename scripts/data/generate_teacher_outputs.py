"""
Generate teacher Tamil translations using the Sarvam AI API (mayura:v1).

Sources:
  irumozhi (default) — 499 IruMozhi English sentences
  flores             — 1012 FLORES-200 devtest English sentences

All modes: API → Tamil Unicode → romanized/normalized Latin for evaluation.

Outputs:
  outputs/teacher/{mode}.jsonl                — IruMozhi source
  outputs/teacher/flores_{mode}.jsonl         — FLORES source
  Each line: {"idx": int, "en": str, "ta": str, "ta_raw": str}

Usage (Colab):
  !python scripts/data/generate_teacher_outputs.py --mode modern-colloquial
  !python scripts/data/generate_teacher_outputs.py --mode modern-colloquial --source flores
"""

import json
import os
import time
import argparse
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from data.tamil_romanizer import clean_sentence
except ModuleNotFoundError:
    from tamil_romanizer import clean_sentence

API_URL = "https://api.sarvam.ai/translate"
OUTPUT_DIR = Path("outputs")
SARVAM_MODES = ["formal", "classic-colloquial", "modern-colloquial"]
FLORES_URL = "https://raw.githubusercontent.com/openlanguagedata/flores/main/flores200_dataset/devtest/devtest.eng_Latn"


def get_api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("SARVAM_API_KEY")
        except Exception:
            pass
    if not key:
        raise ValueError("SARVAM_API_KEY not found in env or Colab Secrets")
    return key


def sarvam_translate(text: str, mode: str, api_key: str, retries: int = 3) -> str:
    headers = {
        "api-subscription-key": api_key,
        "content-type": "application/json",
    }
    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "ta-IN",
        "mode": mode,
        "model": "mayura:v1",
        "speaker_gender": "Male",
        "enable_preprocessing": True,
    }
    for attempt in range(retries):
        r = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json().get("translated_text", "")
    return ""


def load_irumozhi_english() -> list[tuple[int, str]]:
    from datasets import load_dataset
    ds = load_dataset("aryaman/irumozhi", split="train")
    return list(enumerate(ds["english"]))


def load_flores_english() -> list[tuple[int, str]]:
    print("Downloading FLORES-200 devtest English sentences...")
    r = requests.get(FLORES_URL, timeout=30)
    r.raise_for_status()
    sentences = [s.strip() for s in r.text.splitlines() if s.strip()]
    print(f"Loaded {len(sentences)} FLORES sentences")
    return list(enumerate(sentences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=SARVAM_MODES, required=True)
    parser.add_argument("--source", choices=["irumozhi", "flores"], default="irumozhi")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    api_key = get_api_key()
    teacher_dir = OUTPUT_DIR / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    filename = f"flores_{args.mode}.jsonl" if args.source == "flores" else f"{args.mode}.jsonl"
    out_path = teacher_dir / filename

    done: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["idx"])
        print(f"Resuming — {len(done)} examples already done.")

    pairs = load_flores_english() if args.source == "flores" else load_irumozhi_english()
    todo = [(idx, en) for idx, en in pairs if idx not in done]
    print(f"Generating {len(todo)} examples  (source: {args.source}, mode: {args.mode})...")

    with open(out_path, "a", encoding="utf-8") as f:
        for idx, en in tqdm(todo):
            raw = sarvam_translate(en, args.mode, api_key)
            ta = clean_sentence(raw)
            record = {"idx": idx, "en": en, "ta": ta, "ta_raw": raw}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(args.delay)

    print(f"Done. {len(todo)} new examples written to {out_path}")
