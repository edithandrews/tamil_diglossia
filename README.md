# Colloquial Tamil Register Transfer

Code and cached outputs for a final NLP project on fine-tuning mBART-50 to
produce colloquial Tamil translations using synthetic teacher data from the
Sarvam Mayura API.

**Main finding:** LoRA and full fine-tuning reach teacher-level colloquial
register on the IruMozhi test set. IA3 and zero-shot do not.

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

## Reproducing Results

The recommended path is the Colab notebooks. Clone the repo to
`/content/tamil_diglossia` before running them.

**Colab secrets needed:**

| Secret | Required for |
|---|---|
| `GITHUB_TOKEN` | cloning this repo in Colab |
| `HF_TOKEN` | downloading models from HuggingFace |
| `SARVAM_API_KEY` | regenerating teacher outputs (cached outputs already included) |

## Local Commands

```bash
pip install -r requirements.txt

# Prepare Tatoeba train/test split
python data/prepare_tatoeba_splits.py

# Train one condition
python train_student.py --regime lora --size 3000

# Evaluate
python evaluate.py --experiment mixed3000

```

## Key Results Files

Main experiment (mBART, mixed3000 training set):

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
