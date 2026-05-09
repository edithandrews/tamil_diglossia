"""
Smoke test — verifies the environment and data loading before the real run.
Checks imports, dataset access, and tokenizer loading (no full model weights).
"""

import sys

def check(label):
    print(f"  {label}...", end=" ", flush=True)

def ok():
    print("OK")

def fail(e):
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== Smoke Test ===\n")

# --- Imports ---
print("[1] Imports")

check("torch")
try:
    import torch
    ok()
except Exception as e:
    fail(e)

check("transformers")
try:
    import transformers
    ok()
except Exception as e:
    fail(e)

check("datasets")
try:
    import datasets
    ok()
except Exception as e:
    fail(e)

check("peft")
try:
    import peft
    ok()
except Exception as e:
    fail(e)

check("sklearn")
try:
    import sklearn
    ok()
except Exception as e:
    fail(e)

# --- Datasets ---
print("\n[2] Dataset loading")

check("IruMozhi (aryaman/irumozhi)")
try:
    from datasets import load_dataset
    ds = load_dataset("aryaman/irumozhi", split="train")
    assert len(ds) > 0
    print(f"OK ({len(ds)} examples, columns: {ds.column_names})")
except Exception as e:
    fail(e)

check("IruMozhi English column (generation prompts)")
try:
    sample_en = ds[0]["english"]
    sample_lit = ds[0]["tamil"]
    sample_col = ds[0]["colloquial: annotator 1"]
    print(f"OK")
    print(f"    en:         {sample_en}")
    print(f"    literary:   {sample_lit}")
    print(f"    colloquial: {sample_col}")
except Exception as e:
    fail(e)

# --- Pre-trained classifier ---
print("\n[3] Pre-trained IruMozhi classifier")

check("aryaman/xlm-roberta-base-irumozhi")
try:
    from transformers import pipeline
    clf = pipeline("text-classification", model="aryaman/xlm-roberta-base-irumozhi")
    literary   = "சீனிவாசன் தயாரித்தார்."
    colloquial = "seenivaasan thayaarichaar"
    r_lit = clf(literary)[0]
    r_col = clf(colloquial)[0]
    assert r_col["label"] == "LABEL_0", "Expected LABEL_0 for colloquial"
    assert r_lit["label"] == "LABEL_1", "Expected LABEL_1 for literary"
    print(f"OK (literary→{r_lit['label']} {r_lit['score']:.2f}, colloquial→{r_col['label']} {r_col['score']:.2f})")
except Exception as e:
    fail(e)

# --- Tokenizers ---
print("\n[4] Tokenizer loading")

check("IndicBART tokenizer (ai4bharat/IndicBART)")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
    print(f"OK (vocab size: {tok.vocab_size})")
except Exception as e:
    fail(e)

# --- PEFT config smoke test ---
print("\n[5] PEFT config")

check("LoraConfig for seq2seq")
try:
    from peft import LoraConfig, TaskType
    cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=32, lora_dropout=0.1)
    ok()
except Exception as e:
    fail(e)

print("\n=== All checks passed ===\n")
