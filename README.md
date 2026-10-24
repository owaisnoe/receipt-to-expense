# Receipt to Expense

**A fine-tuned Gemma 4 E2B that reads a receipt photo, fills in the expense, and splits the bill.**
Hacktoberfest Hack Day, Theme 3B (receipt extraction with a fine-tuned Gemma 4).
**Team Come_Join_Qarc** (SSIT Tumkur): Mohammed Owais (lead), Mohammed Muzakkir, Mohammad Tariq Afzal, Mohammed Ayaan, Mohammed Younus.

- **Model and model card:** https://huggingface.co/owaisnoe/gemma4-e2b-sroie-lora. Holds the LoRA adapters, full evaluation and eval CSVs.
- **Slides:** [`slides/`](slides). `Theme3B_Receipt_to_Expense_v3.pptx` / `.pdf` is the final deck, with a slide on how we trained the model.

## Results (100 held-out SROIE test receipts)

| Same Gemma 4 E2B, same settings | without our fine-tune | **with our fine-tune** |
|---|---|---|
| Receipts read perfectly (all 4 fields exact) | 7% | **70%** (10x) |
| Fields read correctly | 70% | **92%** (3/4 fewer errors) |
| Receipts fully right vs a hand-engineered prompt | 37% (prompt) | **76%** (2x) |

- McNemar paired test against the engineered prompt: 40 receipts better, 1 worse, p < 1e-10.
- These numbers come from our best configuration (1120 visual tokens), chosen from 5 configurations on the same test set. Our pre-planned first model (560 tokens) already scored 66% vs the prompt's 37% (p = 1e-6).
- Details are in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`results/`](results).

## How it works

- Gemma 4 E2B (an open, 2B-class vision-language model), LoRA r=32 on the vision and language layers (1.15% of the parameters), trained on the 626 SROIE training receipts with Unsloth.
- The image goes before the text in both training and inference. Greedy decoding at batch size 1.
- No OCR step: the model reads the photo directly and outputs JSON (`company`, `date`, `address`, `total`).
- We found and fixed a silent bug. The training collator shrank receipts to 512 px and snapped them to 32-px multiples while inference saw full size. Token counts matched, so we compared pixels; they are now identical (difference 0.0000). See [`notebook/COLAB_NOTES.md`](notebook/COLAB_NOTES.md).

## Repository layout

| Folder | What is in it |
|---|---|
| `slides/` | Final deck (v2, product-landing style) and the earlier FINAL deck, as PPTX and PDF |
| `notebook/` | The Colab notebook template, the cells added on the runtime (`colab_cells/`), and `COLAB_NOTES.md` (what changed and why) |
| `experiments/` | `run_experiment.py`: background runner for E4B, LoRA rank, epochs and visual-budget experiments |
| `app/` | `pro_ui.py`: the polished laptop UI (Scan / Split / Expenses), which calls the Colab app as its GPU backend. Also the logo, sample receipts and tests |
| `docs/` | Model card, dataset description, 3-minute talk, judge Q&A, deliverables checklist, design notes |
| `results/` | Comparison, experiments, learning curve, error categories, trust signal, out-of-distribution and line-item tables and charts |
| `deck/` | Scripts that build the slides, plus their charts |
| `screenshots/` | The app on desktop and phone |

## Running it

1. **Training and evaluation:** open `notebook/theme3_3B_receipts_template.ipynb` in Colab on a G4 or A100 GPU. Run the sections in order, applying the changes in `COLAB_NOTES.md`; the added cells are in `notebook/colab_cells/`.
2. **App in Colab:** `colab_cells/15_app_receipt_to_expense.py` runs the Gradio app with a public share link.
3. **Polished laptop UI:** `pip install gradio==6.30.0 gradio_client pandas pillow`, then `R2E_BACKEND=<the Colab share link> R2E_SHARE=0 python app/pro_ui.py`, and open http://127.0.0.1:7871.

## Data and licences

- **Training data:** SROIE (ICDAR 2019) via `jsdnrs/ICDAR2019-SROIE`, CC BY 4.0. The sample receipts in `app/examples/` come from the SROIE test split.
- **Evaluation-only data:** CORD v2 (CC BY 4.0) and WildReceipt (Sun et al. 2021). Neither was ever trained on.
- **Adapters:** Apache 2.0, the same licence as the Gemma 4 base model.
