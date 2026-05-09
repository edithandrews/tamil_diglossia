"""
Fine-tune a decoder-only causal LM on the Sarvam colloquial Tamil teacher corpus.

This is the runnable version of notebooks/02_decoder_qwen_lora.ipynb.
It keeps the decoder experiment separate from the main mBART seq2seq student
while writing outputs in the same shape expected by evaluate.py.

Usage:
  python scripts/train_decoder_student.py --variant qwen05_mixed3000
  python scripts/train_decoder_student.py --variant qwen15_mixed3000 --generate-only
  python scripts/train_decoder_student.py --variant bloomz_mixed3000 --smoke --overwrite-outputs
  python evaluate.py --experiment decoder_mixed3000_qwen05
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from aksharamukha import transliterate
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


OUTPUT_DIR = Path("outputs")
IRUMOZHI_TRAIN_PATH = OUTPUT_DIR / "data" / "irumozhi_train.jsonl"
IRUMOZHI_TEST_PATH = OUTPUT_DIR / "data" / "irumozhi_test.jsonl"
IRUMOZHI_TEACHER_PATH = OUTPUT_DIR / "teacher" / "modern-colloquial.jsonl"
TATOEBA_TRAIN_PATH = OUTPUT_DIR / "data" / "tatoeba_train.jsonl"
TATOEBA_TEACHER_PATH = OUTPUT_DIR / "teacher" / "tatoeba_modern-colloquial.jsonl"

MAX_LENGTH = 192
MAX_NEW_TOKENS = 80
DEFAULT_TRAIN_SIZE = 3000
SYSTEM_PROMPT = "You translate English into spoken colloquial Tamil. Output only the translation."
TAMIL_RANGE = re.compile(r"[஀-௿]+")

DECODER_EXPERIMENTS = {
    "qwen05_mixed3000": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "experiment_name": "mixed3000_qwen05",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "use_chat_template": True,
    },
    "qwen15_mixed3000": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "experiment_name": "mixed3000_qwen15",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "use_chat_template": True,
    },
    "bloomz_mixed3000": {
        "model_name": "bigscience/bloomz-560m",
        "experiment_name": "mixed3000_bloomz",
        "target_modules": ["query_key_value"],
        "use_chat_template": False,
    },
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def romanize_tamil(text: str) -> str:
    def convert(match):
        return transliterate.process("Tamil", "ISO", match.group(0))

    return TAMIL_RANGE.sub(convert, text)


def make_prompt(en: str, tokenizer, use_chat_template: bool) -> str:
    user_prompt = f"English: {en}\nTamil:"
    if use_chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n{user_prompt}"


def load_training_rows(train_size: int) -> list[dict]:
    if not TATOEBA_TRAIN_PATH.exists():
        subprocess.run([sys.executable, "data/prepare_tatoeba_splits.py"], check=True)

    train_idx = {row["idx"] for row in load_jsonl(IRUMOZHI_TRAIN_PATH)}
    irumozhi_rows = [row for row in load_jsonl(IRUMOZHI_TEACHER_PATH) if row["idx"] in train_idx]
    irumozhi_rows.sort(key=lambda row: row["idx"])

    n_tatoeba = max(0, train_size - len(irumozhi_rows))
    tatoeba_train_rows = load_jsonl(TATOEBA_TRAIN_PATH)[:n_tatoeba]
    tatoeba_teacher_lookup = {row["idx"]: row for row in load_jsonl(TATOEBA_TEACHER_PATH)}
    tatoeba_rows = [
        tatoeba_teacher_lookup[row["idx"]]
        for row in tatoeba_train_rows
        if row["idx"] in tatoeba_teacher_lookup
    ]

    print(f"Training data: {len(irumozhi_rows)} IruMozhi + {len(tatoeba_rows)} Tatoeba = {len(irumozhi_rows) + len(tatoeba_rows)} total")
    return irumozhi_rows + tatoeba_rows


def experiment_output_dir(config: dict) -> Path:
    return OUTPUT_DIR / "decoder" / config["experiment_name"] / "student_lora"


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )


def load_trained_model(config: dict, out_dir: Path):
    final_model = out_dir / "final_model"
    if not final_model.exists():
        raise FileNotFoundError(f"No trained decoder adapter found at {final_model}")
    base = load_base_model(config["model_name"])
    return PeftModel.from_pretrained(base, str(final_model))


def tokenize_rows(rows: list[dict], tokenizer, use_chat_template: bool) -> Dataset:
    eos = tokenizer.eos_token or ""

    def tokenize_one(row: dict) -> dict:
        prompt = make_prompt(row["en"], tokenizer, use_chat_template)
        target = " " + row["ta"].strip() + eos
        encoded = tokenizer(
            prompt + target,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        prompt_len = min(len(prompt_ids), MAX_LENGTH)
        labels = encoded["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        labels = [label if mask else -100 for label, mask in zip(labels, encoded["attention_mask"])]
        encoded["labels"] = labels
        return encoded

    return Dataset.from_list([tokenize_one(row) for row in rows])


def train_decoder(config: dict, out_dir: Path, args) -> None:
    tokenizer = load_tokenizer(config["model_name"])
    model = load_base_model(config["model_name"])
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    rows = load_training_rows(args.train_size)
    dataset = tokenize_rows(rows, tokenizer, config["use_chat_template"])

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=1,
        max_steps=2 if args.smoke else -1,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")


def generate_outputs(config: dict, out_dir: Path, overwrite: bool = False) -> None:
    output_path = out_dir / "outputs.jsonl"
    if output_path.exists() and not overwrite:
        print(f"Skipping generation because {output_path} already exists. Use --overwrite-outputs to replace it.")
        return

    tokenizer = load_tokenizer(config["model_name"])
    model = load_trained_model(config, out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_rows = load_jsonl(IRUMOZHI_TEST_PATH)
    test_rows.sort(key=lambda row: row["idx"])
    results = []
    with torch.no_grad():
        for row in tqdm(test_rows, desc="generating"):
            prompt = make_prompt(row["en"], tokenizer, config["use_chat_template"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append({
                "source": "irumozhi",
                "idx": row["idx"],
                "en": row["en"],
                "generated": romanize_tamil(generated),
            })

    with output_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} outputs to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(DECODER_EXPERIMENTS), default="qwen05_mixed3000")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run two training steps before generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = DECODER_EXPERIMENTS[args.variant]
    out_dir = experiment_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.generate_only:
        train_decoder(config, out_dir, args)
    if not args.skip_generate:
        generate_outputs(config, out_dir, overwrite=args.overwrite_outputs)
