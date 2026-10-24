# Colab run notes (24 Oct 2026, Colab G4: RTX PRO 6000 Blackwell, 95 GB)

The template notebook (`theme3_3B_receipts_template.ipynb`) was run section by section through colab-mcp. Below: what was measured at each gate, and every change we made on the runtime. The cells we added are in `colab_cells/`.

## Environment
torch 2.11.0+cu130, transformers 5.5.0, unsloth / unsloth_zoo 2026.10.8, trl 1.16.0, peft 0.21.2, datasets 4.3.0, gradio 6.30.0, Python 3.13.
The only pip conflict was `gcsfs` vs `fsspec`, which the pipeline doesn't use.

## Gates
| Gate | Result |
|---|---|
| B: visual budget | `images_kwargs={"max_soft_tokens": 560}` works: 525-544 image tokens. Output is JSON. Steady state about 2.4-3.0 s per receipt; the first call takes 20 s of warm-up. |
| B2: training vs inference | Token counts matched, but **pixels did not** (see fix 1). After the fix, the pixel difference is 0.0000 on receipts from 463 px to 4961 px wide. |
| C: labels | The instruction and the JSON answer are trained; image tokens are masked (119 trained tokens). |
| D: smoke training | Losses finite: 2.56 → 1.16. Steady state 0.96 s/step; 157 steps took 2.5 min, peak 13 GB. |
| SDPA bug check | 0/100 prompts have length 1 mod 32 (both instructions); eager attention matched SDPA exactly on 3 + 3 receipts. |

## Changes made on the runtime
1. **Collator pixel fix.** `UnslothVisionDataCollator` defaults to `resize="min"`, which shrinks images to 512 px wide. Images embedded in messages also get snapped to 32-px multiples.
   - Fix: put a bare `{"type": "image"}` placeholder first in the message, pass the image in an `images` column, and build the collator with `UnslothVisionDataCollator(model, processor, resize="max")`.
2. **Local receipts** (section 14): added `ImageOps.exif_transpose` so phone photos are upright, and `pd.read_csv(..., dtype=str)` so a total like `9.00` doesn't turn into `9.0`.
3. **`GEN_LOCK`** (an RLock): `model.disable_adapter()` acts globally, so every "set adapter + generate" call takes the lock. That way the live app and the evaluation cells can't mix fine-tuned and plain outputs. `evaluate()` was redefined to take the lock; its outputs are unchanged.
4. **Background experiments in the same VM:** `experiments/run_experiment.py` ran 3 parallel queues on the shared GPU (E4B; r=8 and 2 epochs; budgets 1120 and 280) with per-process memory caps.
5. **Name-collision bug (found late):** a results cell reused the name `_rows`, which broke the app's "Fill items" and "By items". `colab_cells/99_fix_best_model_fast_endpoint.py` restores the helper. It also swaps in the best (1120) adapter in place and adds a fast fine-tuned-only endpoint.
6. **HF publishing:** the adapters were uploaded exactly as saved (never merged), followed by the filled model card and the evaluation files.

## Main results (all four fields correct, normalized / strict)
- **560 budget (pre-planned):** plain 28 / 6, prompted 37 / 17, fine-tuned 66 / 63.
- **1120 budget (best):** fine-tuned 76 / 70, with field accuracy 92% (company 93, date 98, address 80, total 98).
- **Other experiments:** E4B 69 / 64; r=8 68 / 64; 2 epochs 71 / 67; budget 280 61 / 56.
- **Generalisation:** 63% of test receipts come from shops seen in training. On new shops the fine-tuned model scores 37.8 vs 18.9 for the prompted model. The out-of-distribution sets (WildReceipt, CORD) show no significant loss.

## Not included
The executed Colab notebook itself is not here, because the editing bridge disconnected near the end. To add it, use File > Download > .ipynb from the Colab tab and save it as `notebook/theme3_3B_receipts_executed.ipynb`.
