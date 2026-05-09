# Colloquial Tamil Register Transfer

Code and cached outputs for a final NLP project (COSI 115b, Brandeis University)
on fine-tuning mBART-50 to produce colloquial Tamil translations using synthetic
teacher data from the Sarvam Mayura API.

**Main finding:** LoRA and full fine-tuning reach teacher-level colloquial
register on the IruMozhi test set. IA3 and zero-shot do not.

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/edithandrews/tamil_diglossia.git
cd tamil_diglossia
pip install -r requirements.txt
```

## Reproducing Results

The recommended path is to run the Colab notebooks, which handle GPU setup and
dependencies automatically.

**Before running, set these Colab secrets:**

| Secret | Required for |
|---|---|
| `GITHUB_TOKEN` | cloning this repo in Colab |
| `HF_TOKEN` | downloading models from HuggingFace |
| `SARVAM_API_KEY` | regenerating teacher outputs (cached outputs already included, so this is optional) |

**Notebooks:**

- `notebooks/01_mbart_training_eval.ipynb` — main reproducibility runbook.
  Runs all mBART fine-tuning conditions (full, LoRA, QLoRA, IA3, zero-shot)
  and evaluates them on the IruMozhi test set. Each cell checks for existing
  outputs before training so nothing is re-run unnecessarily.

- `notebooks/02_decoder_qwen_lora.ipynb` — decoder-only Qwen LoRA comparison
  (exploratory).

The notebooks expect the repo to be cloned at `/content/tamil_diglossia`.

## Results

Main experiment results (mBART, mixed3000 training set):

```text
outputs/mbart/mixed3000/results.csv
```

Supporting experiments:

```text
outputs/mbart/irumozhi399/results.csv
outputs/mbart/mixed1000/results.csv
outputs/mbart/mixed2000/results.csv
outputs/decoder/irumozhi399_qwen05/results.csv
outputs/decoder/mixed3000_qwen05/results.csv
```

## Repo Structure

```text
.
├── evaluate.py                  # evaluation script (classifier + chrF/BLEU)
├── train_student.py             # mBART fine-tuning (full / LoRA / QLoRA / IA3)
├── requirements.txt
│
├── data/                        # data utilities used by notebooks
│   ├── prepare_tatoeba_splits.py
│   └── tamil_romanizer.py
│
├── notebooks/
│   ├── 01_mbart_training_eval.ipynb   # main mBART training + evaluation runbook
│   └── 02_decoder_qwen_lora.ipynb     # decoder-only Qwen LoRA comparison
│
├── scripts/                     # helper scripts (not required to reproduce)
│   ├── classify_teacher_outputs.py
│   ├── smoke_test.py
│   ├── train_decoder_student.py
│   ├── audit_review_app.py
│   └── data/                    # data prep and teacher generation
│
└── outputs/
    ├── data/                    # train/test splits
    ├── teacher/                 # cached Sarvam teacher translations
    ├── mbart/                   # mBART outputs and results CSVs
    ├── decoder/                 # decoder-only outputs and results CSVs
    └── audit/                   # qualitative adequacy audit files
```

Training checkpoints and final model weights are not included
(`outputs/**/checkpoint-*` and `outputs/**/final_model/` are gitignored).
