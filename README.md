# Colloquial Tamil Register Distillation

This repository contains the runnable code and cached outputs for a final NLP
project on transferring colloquial Tamil register from a commercial teacher model
to smaller open models.

The main result is from mBART-50 fine-tuning on Sarvam Mayura teacher outputs.
On the 100-example held-out IruMozhi test set, LoRA and full fine-tuning reach
teacher-level colloquial register, while IA3 and zero-shot remain much weaker.

The paper source is currently local under `writeup/final_paper.tex`. The writeup
files are intentionally left out of the GitHub upload for now.

## What To Run

The easiest way to reproduce the project is to run the Colab notebooks:

- `notebooks/01_mbart_training_eval.ipynb`  
  Main mBART training/evaluation runbook.

- `notebooks/02_decoder_qwen_lora.ipynb`  
  Decoder-only Qwen LoRA comparison.

The notebooks expect the repo to be cloned at:

```text
/content/tamil_diglossia
```

For the private GitHub repo, set these Colab secrets before running:

```text
GITHUB_TOKEN
SARVAM_API_KEY
HF_TOKEN
```

`SARVAM_API_KEY` is only needed if regenerating teacher outputs. Cached teacher
outputs are already included under `outputs/teacher/`.

## Core Files

These Python files are required by the notebooks:

```text
evaluate.py
train_student.py
data/prepare_tatoeba_splits.py
data/tamil_romanizer.py
```

Optional helper scripts are in `scripts/`. They are useful for diagnostics,
audit work, teacher regeneration, and figure generation, but the notebooks do
not depend on most of them.

## Data And Outputs

Important cached artifacts:

```text
outputs/data/       input data and train/test splits
outputs/teacher/    cached Sarvam teacher translations
outputs/mbart/      main mBART outputs and evaluation CSVs
outputs/decoder/    decoder-only outputs and evaluation CSVs
outputs/audit/      qualitative adequacy audit files
```

Large training checkpoints are not included. The `.gitignore` excludes:

```text
outputs/**/checkpoint-*
outputs/**/final_model/
```

## Local Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the Tatoeba train/test split:

```bash
python data/prepare_tatoeba_splits.py
```

Run one mBART training condition:

```bash
python train_student.py --regime lora --size 3000
```

Evaluate the main experiment:

```bash
python evaluate.py --experiment mixed3000
```

Regenerate the paper figure locally:

```bash
python scripts/writeup/generate_results_figure.py
```

## Main Results File

The headline results are stored here:

```text
outputs/mbart/mixed3000/results.csv
```

Supporting result files:

```text
outputs/mbart/irumozhi399/results.csv
outputs/mbart/mixed1000/results.csv
outputs/mbart/mixed2000/results.csv
outputs/decoder/irumozhi399_qwen05/results.csv
outputs/decoder/mixed3000_qwen05/results.csv
```