"""
Fine-tune mBART-50 on the teacher-generated colloquial corpus.

Training regimes: full fine-tuning, LoRA, QLoRA (4-bit LoRA), IA3 adapter, zero-shot baseline.
Dataset sizes:    500 | 1000 | 2000 | 3000
                  Always 399 IruMozhi train + (size-399) Tatoeba train examples.
Data modes:       mixed       — IruMozhi train + Tatoeba augmentation, IruMozhi test
                  irumozhi399 — IruMozhi train only, IruMozhi test only

Usage:
  python train_student.py --regime lora --size 1000
  python train_student.py --regime lora --data irumozhi399
  python train_student.py --regime qlora --size 2000
  python train_student.py --regime adapter --size 3000
  python train_student.py --regime full --size 3000
  python train_student.py --regime baseline_zeroshot
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from data.tamil_romanizer import aksharamukha_sentence, clean_sentence
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import get_peft_model, PeftModel, LoraConfig, IA3Config, TaskType
from transformers.trainer_utils import get_last_checkpoint

STUDENT_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TAMIL_TOKEN = "ta_IN"
OUTPUT_DIR = Path("outputs")
REGIMES = ["full", "lora", "qlora", "adapter", "baseline_zeroshot"]
VALID_SIZES = [500, 1000, 2000, 3000]
DATA_MODES = ["mixed", "irumozhi399"]

IRUMOZHI_TEACHER_PATH = OUTPUT_DIR / "teacher" / "modern-colloquial.jsonl"
IRUMOZHI_TRAIN_PATH   = OUTPUT_DIR / "data" / "irumozhi_train.jsonl"
IRUMOZHI_TEST_PATH    = OUTPUT_DIR / "data" / "irumozhi_test.jsonl"
TATOEBA_TRAIN_PATH    = OUTPUT_DIR / "data" / "tatoeba_train.jsonl"
TATOEBA_TEST_PATH     = OUTPUT_DIR / "data" / "tatoeba_test.jsonl"
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128


def _load_model(model_path: str | None = None, quantize_4bit: bool = False):
    model_name_or_path = model_path or STUDENT_MODEL
    if quantize_4bit:
        import torch
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, quantization_config=bnb_config)
    return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


def _load_tokenizer():
    return AutoTokenizer.from_pretrained(
        STUDENT_MODEL, src_lang=SRC_LANG, tgt_lang=TAMIL_TOKEN
    )


def load_corpus(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_training_data(size: int) -> list[dict]:
    """399 IruMozhi train examples (fixed) + first (size-399) Tatoeba train examples."""
    train_idx = {json.loads(l)["idx"] for l in open(IRUMOZHI_TRAIN_PATH)}
    irumozhi = [r for r in load_corpus(IRUMOZHI_TEACHER_PATH) if r["idx"] in train_idx]

    n_tatoeba = size - len(irumozhi)
    tatoeba = load_corpus(TATOEBA_TRAIN_PATH)[:n_tatoeba]

    print(f"Training data: {len(irumozhi)} IruMozhi + {len(tatoeba)} Tatoeba = {len(irumozhi) + len(tatoeba)} total")
    return irumozhi + tatoeba


def load_irumozhi399_training_data() -> list[dict]:
    """IruMozhi-only distillation corpus: 399 train examples, no Tatoeba augmentation."""
    train_idx = {json.loads(l)["idx"] for l in open(IRUMOZHI_TRAIN_PATH)}
    irumozhi = [r for r in load_corpus(IRUMOZHI_TEACHER_PATH) if r["idx"] in train_idx]
    irumozhi.sort(key=lambda r: r["idx"])
    print(f"Training data: {len(irumozhi)} IruMozhi only")
    return irumozhi


def experiment_output_dir(regime: str, data_mode: str, size: int | None = None) -> Path:
    if data_mode == "irumozhi399":
        return OUTPUT_DIR / "mbart" / "irumozhi399" / f"student_{regime}"
    if size is not None:
        return OUTPUT_DIR / "mbart" / f"mixed{size}" / f"student_{regime}"
    return OUTPUT_DIR / "mbart" / f"student_{regime}"


def tokenize_corpus(corpus: list[dict], tokenizer) -> Dataset:
    tokenized = tokenizer(
        [ex["en"] for ex in corpus],
        text_target=[ex["ta_raw"] for ex in corpus],
        max_length=MAX_SRC_LEN,
        truncation=True,
    )
    pad_id = tokenizer.pad_token_id
    tokenized["labels"] = [
        [-100 if t == pad_id else t for t in label]
        for label in tokenized["labels"]
    ]
    return Dataset.from_dict(tokenized)


def load_trained_model(out_dir: Path, regime: str):
    final_model = out_dir / "final_model"
    if final_model.exists():
        if (final_model / "adapter_config.json").exists():
            return PeftModel.from_pretrained(_load_model(), str(final_model))
        return _load_model(str(final_model))
    checkpoint = get_last_checkpoint(str(out_dir))
    if checkpoint is None:
        raise FileNotFoundError(f"No trained model found in {out_dir}")
    if regime in ("lora", "qlora", "adapter"):
        return PeftModel.from_pretrained(_load_model(), checkpoint)
    return _load_model(checkpoint)


def get_peft_config(regime: str, lora_rank: int = 32, lora_alpha: int | None = None):
    if regime in ("lora", "qlora"):
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_rank,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
    if regime == "adapter":
        return IA3Config(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["k_proj", "v_proj", "fc2"],
            feedforward_modules=["fc2"],
        )
    return None


def regime_label(regime: str, lora_rank: int) -> str:
    """Output-folder label. r=32 keeps the canonical 'lora' name; other ranks get a suffix."""
    if regime == "lora" and lora_rank != 32:
        return f"lora_r{lora_rank}"
    return regime


def train(
    regime: str,
    size: int | None,
    data_mode: str = "mixed",
    smoke: bool = False,
    include_tatoeba_test: bool = False,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
):
    label = regime_label(regime, lora_rank)
    out_dir = experiment_output_dir(label, data_mode, size)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer()

    if regime == "baseline_zeroshot":
        model = _load_model()
        generate_outputs(model, tokenizer, out_dir, include_tatoeba=include_tatoeba_test)
        return

    model = _load_model(quantize_4bit=(regime == "qlora"))

    if regime == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    peft_config = get_peft_config(regime, lora_rank=lora_rank, lora_alpha=lora_alpha)
    if peft_config:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    corpus = load_irumozhi399_training_data() if data_mode == "irumozhi399" else load_training_data(size)
    train_dataset = tokenize_corpus(corpus, tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=True,
        predict_with_generate=True,
        save_strategy="epoch",
        save_total_limit=1,
        max_steps=2 if smoke else -1,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    checkpoint = get_last_checkpoint(str(out_dir))
    trainer.train(resume_from_checkpoint=checkpoint)

    model.save_pretrained(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")

    generate_outputs(model, tokenizer, out_dir, include_tatoeba=include_tatoeba_test)


def generate_outputs(model, tokenizer, out_dir: Path, include_tatoeba: bool = True):
    import torch
    ta_token_id = tokenizer.lang_code_to_id[TAMIL_TOKEN]

    test_corpus = []
    for ex in load_corpus(IRUMOZHI_TEST_PATH):
        test_corpus.append({**ex, "source": "irumozhi"})
    if include_tatoeba and TATOEBA_TEST_PATH.exists():
        for ex in load_corpus(TATOEBA_TEST_PATH):
            test_corpus.append({**ex, "source": "tatoeba"})

    model.eval()
    results = []
    with torch.no_grad():
        for ex in tqdm(test_corpus, desc="generating"):
            inputs = tokenizer(
                ex["en"],
                return_tensors="pt",
                max_length=MAX_SRC_LEN,
                truncation=True,
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=ta_token_id,
                max_new_tokens=MAX_TGT_LEN,
                max_length=None,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            generated_tamil = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({
                "source": ex["source"],
                "idx": ex["idx"],
                "en": ex["en"],
                "generated_tamil": generated_tamil,
                "generated_iso": aksharamukha_sentence(generated_tamil),
                "generated": clean_sentence(generated_tamil),
            })

    out_path = out_dir / "outputs.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} outputs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--size", type=int, choices=VALID_SIZES,
                        help="Training dataset size for --data mixed (not required for baseline_zeroshot or irumozhi399)")
    parser.add_argument("--data", choices=DATA_MODES, default="mixed",
                        help="Training/evaluation data mode")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--include-tatoeba-test", action="store_true",
                        help="Also generate/evaluate Tatoeba test rows. Default is IruMozhi test only.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 training steps only — sanity check")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank r (default 32). Other values write to student_lora_r{rank}/.")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default = 2 * rank).")
    args = parser.parse_args()

    if args.data == "mixed" and args.regime != "baseline_zeroshot" and args.size is None:
        parser.error("--size is required for mixed data runs except baseline_zeroshot")

    if args.generate_only:
        import torch
        label = regime_label(args.regime, args.lora_rank)
        out_dir = experiment_output_dir(label, args.data, args.size)
        tokenizer = _load_tokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_trained_model(out_dir, args.regime).to(device)
        generate_outputs(model, tokenizer, out_dir, include_tatoeba=args.include_tatoeba_test)
    else:
        train(
            args.regime,
            args.size,
            data_mode=args.data,
            smoke=args.smoke,
            include_tatoeba_test=args.include_tatoeba_test,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
