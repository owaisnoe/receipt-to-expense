---
base_model: unsloth/gemma-4-E2B-it
library_name: peft
license: apache-2.0
datasets:
- jsdnrs/ICDAR2019-SROIE
language:
- en
pipeline_tag: image-text-to-text
tags:
- gemma4
- lora
- unsloth
- vision-language
- receipts
- key-information-extraction
- sroie
- hacktoberfest
---

# Gemma 4 E2B receipt extractor (LoRA adapters, SROIE)

LoRA adapters for [`unsloth/gemma-4-E2B-it`](https://huggingface.co/unsloth/gemma-4-E2B-it) that read a photo or scan of a shop receipt and return one JSON object with four fields: `company`, `date`, `address` and `total`.

They were built at Hacktoberfest Hack Day (Theme 3B, 24 Oct 2026) as the model behind a small "receipt to expense" app. The evaluation CSVs, charts and settings are in the [`eval/`](./eval) folder of this repository.

> **Status:** hackathon research artefact. Every number below was measured in one run on a Colab G4 GPU; the main test set is 100 SROIE test receipts.

## Results at a glance

| All four fields correct (100 SROIE test receipts) | Plain | Prompted | **Fine-tuned (these adapters)** |
|---|---|---|---|
| Strict (exact strings) | 6 % | 17 % | **63 %** |
| Normalized (case, spacing, currency, date format forgiven) | 28 % | 37 % | **66 %** |

- Fine-tuned vs the detailed-prompt baseline: 33 receipts improved, 4 got worse (normalized), exact McNemar p = 1.1e-06.
- **Read this with the vendor caveat:** 63 of the 100 test receipts come from shops that also appear in the training labels. On those, the fine-tuned model gets all four fields right 82.5 % of the time; on the 37 unseen shops, 37.8 % (prompted baseline: 18.9 %).
- Out of domain (WildReceipt, CORD), fine-tuning is neutral: no significant gain or loss.

## Intended use

- **Users:** students and small shops logging expenses. Photograph a receipt, check the four fields, then add them to an expense table or CSV.
- **Input:** one receipt image per call. The prompt puts the image first, followed by the training instruction (below).
- **Output:** a JSON object in SROIE's conventions:
  - company and address in capitals, as printed;
  - date exactly as printed (the format is not normalised);
  - total as a bare number such as `9.00`, without a currency symbol.
- **Keep a human in the loop.** The demo app flags "please check" when the date or total does not parse. Do not use the output for accounting, tax or audit without review.
- **Out of scope:** handwriting; non-Latin scripts; multi-receipt images; any decision about a person. Line items were only tested zero-shot (see below).

## Training data

- **Dataset:** [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE), a Hugging Face packaging of the ICDAR 2019 SROIE competition data (Huang et al., 2019, [arXiv:2103.10213](https://arxiv.org/abs/2103.10213)), Task 3 "Key Information Extraction from Scanned Receipts".
- **Licence:** CC BY 4.0 according to the dataset card.
- **Data used:** all **626** training receipts. Each training example is the receipt image plus the instruction, with the gold JSON (four fields) as the answer.
- **Never trained on:** any of the **361** SROIE test receipts, and none of the external receipts used for the robustness checks (WildReceipt, CORD).

## Training procedure

| Setting | Value |
|---|---|
| Base model | `unsloth/gemma-4-E2B-it`, loaded in 16-bit (`load_in_4bit=False`) |
| Method | LoRA via Unsloth `FastVisionModel` (16-bit LoRA, adapters only, never merged) |
| LoRA | r = 32, alpha = 32, dropout 0, bias none; vision **and** language layers; attention and MLP modules; `random_state=3407` |
| Trainable | 59.8 M of 5.18 B parameters (1.15 %); adapter file 239.5 MB |
| Optimiser | `adamw_8bit`, learning rate 2e-4, cosine schedule, warm-up ratio 0.03, weight decay 0.001, max grad norm 0.3 |
| Batch | 1 per device x gradient accumulation 4 (effective 4) |
| Steps | 157 = 1 epoch over 626 receipts. A 5-step smoke run on the same adapters came first, so the adapters received 5 + 157 updates. |
| Sequence | `max_length=2048`; SFTConfig vision settings `remove_unused_columns=False`, `dataset_text_field=""`, `dataset_kwargs={"skip_prepare_dataset": True}` |
| Visual budget | `max_soft_tokens = 560` (Gemma 4 offers 70/140/280/560/1120); **measured 525-544 image tokens per receipt** |
| Prompt layout | image first, then text (as the Gemma 4 model card recommends), in both training and inference |
| Instruction | `Extract the company, date, address and total from this receipt. Reply with JSON only.` |
| Target | `{"company": ..., "date": ..., "address": ..., "total": ...}` with the SROIE gold strings unchanged |
| Loss masking | the instruction and the JSON answer are trained; image tokens are masked (checked on a collated batch) |
| Collator | `UnslothVisionDataCollator(model, processor, resize="max")`, with images passed as an `images` column (see the fix below) |
| Chat template | `unsloth.chat_templates.get_chat_template(processor, "gemma-4")`, saved with the adapter |
| Decoding (eval) | greedy (`do_sample=False`), batch size 1, `max_new_tokens=256` |
| Hardware | Colab G4: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB), bf16, SDPA attention (FlashAttention 2 is unsupported for head dim 512) |
| Software | torch 2.11.0+cu130, transformers 5.5.0, unsloth 2026.10.8, unsloth_zoo 2026.10.8, trl 1.16.0, peft 0.21.2, accelerate 1.15.0, datasets 4.3.0, Python 3.13 |
| Cost | 157 steps in 2.5 min (0.96 s/step including about a minute of compile warm-up); peak GPU memory 13.0 GB; model alone after load: 10.3 GB |
| Loss | smoke steps 1-5: 2.56 to 1.16; main run: 0.98 at step 5, 0.016 at step 155 |

### Two problems found and fixed on the runtime

1. **Training pixels differed from inference pixels (fixed).**
   - The training collator and inference produced the **same number** of image tokens (for example 525 and 525), but different pixels:
     - by default, `UnslothVisionDataCollator` shrinks every receipt to **512 px wide** (`resize="min"`, because the model config has no default image size);
     - for images embedded in the messages, it also snaps width and height to multiples of **32 px** (`patch_size*2`) before the Gemma processor applies the 560-token budget.
   - The measured maximum absolute difference in `pixel_values` was 0.34 on a 463x1013 receipt and 0.68 on a 4961x7016 receipt.
   - In other words, the model would have trained on blurrier, rescaled receipts than it reads at test time, and a token-count check alone would not have caught this.
   - **Fix:** put a `{"type": "image"}` placeholder (still first) in the message, pass the image in an `images` column, and build the collator with `resize="max"`. The collator then loads the column image without snapping or shrinking.
   - **Result:** identical `pixel_values` (**max difference 0.0000**) and identical prompt token IDs on all six receipts checked, from 463 px to 4961 px wide.
2. **Gold-label noise (documented, not fixed).**
   - Some SROIE gold strings carry OCR slips. In one test receipt, the gold address has the postcode **`B1750`**, while both the plain and the fine-tuned model read **`81750`**, which is the real postcode of Masai, Johor.
   - Strict scoring counts such a case as an error for every model. We did not edit any labels, so all reported scores include this noise.

## Evaluation

**Test set.**
- 100 receipts drawn once from the 361-receipt SROIE test split, using `random.Random(0)` (seed 0).
- Their keys are saved in `eval/test_keys.json`, and the code asserts that no test key appears in the training split.

**Configurations.** All three share the same 100 receipts, the image-first layout, the 560 budget, 16-bit loading and greedy batch-1 decoding.
- **Plain:** the base model with the short instruction above.
- **Prompted:** the base model with a detailed instruction that spells out SROIE's conventions (capitals, date as printed, bare total). This shows what prompting alone achieves.
- **Fine-tuned:** these adapters with the short instruction.

**Metrics.**
- *Strict*: the exact string after trimming.
- *Normalized*: company and address upper-cased, whitespace collapsed, edge punctuation stripped; the total parsed as a number to 2 d.p., ignoring currency; the date parsed day-first and compared as a calendar date.
- *All four*: every field on the receipt is correct (the headline).
- *Valid JSON*: the output parses as a JSON object.

| % of 100 receipts | Plain | Prompted | Fine-tuned |
|---|---|---|---|
| Valid JSON | 100 | 100 | 100 |
| Company strict / norm | 68 / 76 | 83 / 85 | **91 / 92** |
| Date strict / norm | 81 / 99 | 82 / 98 | **99 / 99** |
| Address strict / norm | 24 / 52 | 41 / 63 | **71 / 72** |
| Total strict / norm | 62 / 71 | 63 / 73 | **92 / 95** |
| **All four, strict** | 6 | 17 | **63** |
| **All four, normalized** | 28 | 37 | **66** |

**Paired significance** (exact McNemar on per-receipt all-four correctness; b = receipts that became correct, w = receipts that became wrong):
- fine-tuned vs plain: normalized b = 42, w = 4, p = 5.1e-09; strict b = 60, w = 3, p = 9e-15
- fine-tuned vs prompted: normalized b = 33, w = 4, p = 1.1e-06; strict b = 48, w = 2, p = 2.3e-12

**Speed (G4, greedy, batch 1, idle GPU, mean over 100 receipts).** Plain 2.4 s, prompted 2.0 s, fine-tuned 3.0 s per receipt. The adapters are not merged, so the LoRA layers add about 25 % per receipt.

**Seen vs unseen shops.** 63 of the 100 test receipts come from a (company, address) pair that also appears in the training labels (similarity at least 0.9 on both fields, same numbers). All four fields correct, normalized:

| | Seen shops (n = 63) | Unseen shops (n = 37) |
|---|---|---|
| Plain | 34.9 % | 16.2 % |
| Prompted | 47.6 % | 18.9 % |
| **Fine-tuned** | **82.5 %** | **37.8 %** |

So part of the headline is memorised vendors; on genuinely new shops the gain is smaller but still about double the prompted baseline.

**Remaining errors of the fine-tuned model** (100 receipts, one category per field):

| Category | Company | Date | Address | Total |
|---|---|---|---|---|
| correct | 91 | 99 | 71 | 92 |
| format only (normalized match) | 1 | 0 | 1 | 3 |
| punctuation / spacing only | 3 | 0 | 9 | 0 |
| near miss (1-2 characters, similarity at least 0.9) | 2 | 0 | 11 | 0 |
| OCR-confusable characters (8/B, 0/O...) | 0 | 0 | 3 | 0 |
| extra text (e.g. a phone number appended) | 2 | 0 | 1 | 0 |
| truncated | 0 | 0 | 1 | 0 |
| rounding adjustment (within 0.05) | 0 | 0 | 0 | 3 |
| different value | 1 | 1 | 3 | 2 |

Address is the weak field: about 12 of its 28 errors are punctuation conventions or label noise, and most of the rest are one- or two-character reading slips. Totals improved the most (plain: 13 missing and 13 wrong amounts; fine-tuned: 2 wrong amounts and 3 Malaysian rounding adjustments).

**Comparability.** Published SROIE results (for example LayoutLM-family models at about 95-98 entity F1) use OCR tokens and entity F1 on the official split. They are **not** directly comparable to end-to-end generation scored by exact match on this 100-receipt subset.

## Robustness and bonus experiments

- **Out-of-distribution receipts (never trained on), same instruction and settings:**
  - WildReceipt (50 receipts, seed 0; US/EU/Asian photos; labelled fields only): all labelled fields 58 % plain vs 56 % fine-tuned (p = 1.0); total 87.5 vs 93.8 % (p = 0.25); company 72 vs 68 %.
  - CORD v2 (50 Indonesian receipts, total only): 87.5 vs 91.7 % (p = 0.62).
  - Reading: the large in-domain gain does not transfer to other receipt styles, but nothing gets significantly worse, and the total moves in the same direction on all three datasets.
- **Retrieval-augmented variant (vendor memory, train labels only, threshold 0.85 fixed in advance):** fine-tuned 66 to 69 % all-four normalized (3 improved, 0 worse, p = 0.25); plain 28 to 39 % and prompted 37 to 47 % (both significant). Retrieval mostly recovers what fine-tuning already learned about repeat vendors.
- **Line items (zero-shot, 50 CORD receipts, 126 item lines):** item F1 71.0 % plain vs 78.8 % fine-tuned; receipts with every item right 54 vs 64 % (p = 0.3). No forgetting, but the fine-tuned model writes item names in SROIE-style capitals and once shifted prices by one row on a long receipt.
- **Learning curve (checkpoints of the main run, first 50 test receipts, all four normalized / strict):** step 0 (plain) 26 / 6; step 40 (160 receipts seen) 54 / 46; step 80 52 / 48; step 120 54 / 52; step 157 54 / 52. Nearly all of the gain arrives in the first quarter epoch. These are snapshots of one run with a cosine schedule, n = 50.
- **Training experiments** (each arm changes one setting and is trained and tested on its own; all four normalized / strict on the same 100 receipts; exact McNemar against these adapters, 66 / 63):
  - LoRA rank 8: 68 / 64 (p = 0.69). Rank 32 is not needed.
  - 2 epochs (314 steps): 71 / 67 (p = 0.18).
  - Visual budget 280: 61 / 56 (p = 0.38).
  - **Visual budget 1120: 76 / 70 (11 receipts improved, 1 got worse, p = 0.006).** Resolution is the setting that matters.
  - The plain model does not benefit from more visual tokens (all four normalized: 33 % at 280, 28 % at 560, 22 % at 1120).
  - The 1120 run is the best configuration we found. The adapters published here are the pre-planned 560 run, so that the headline is not the best of several runs on the same test set.
- **Model size, E4B vs E2B** (same recipe): plain E4B 42 / 16, prompted 44 / 32, fine-tuned E4B 69 / 64, vs fine-tuned E2B 66 / 63 (9 improved, 6 worse, p = 0.61). Fine-tuned E4B runs at 3.46 s per receipt with a 17.4 GB peak (idle GPU, 20 receipts), vs 3.0 s and about 11 GB for E2B. The doubled model is slower and heavier but not significantly better; a higher visual budget helps E2B more.
- **Trust signal (used in the app):** when the fine-tuned and plain models agree on a field, the fine-tuned value was right 97 % (company), 100 % (date), 83 % (address) and 96 % (total) of the time. Disagreement flags 75 %, 100 %, 64 % and 40 % of the fine-tuned errors respectively. Useful but imperfect, so the app says "double-check", never "verified".
- **Demo app:** a Gradio "Receipt to Expense" app with the fine-tuned and plain models side by side, a known-vendor badge, zero-shot line items, split-the-bill with friends (equally or by item, tax shared in proportion, fewest-payments settle-up) and an expense table with CSV export.

## Limitations and biases

- **Narrow domain.**
  - SROIE is a set of English-language Malaysian shop receipts (Johor and Selangor addresses, RM amounts, GST lines, printed dates around 2016-2019 in the receipts we inspected).
  - Expect lower accuracy on other countries, currencies, languages, thermal-paper fading, long receipts, photos taken at an angle and handwriting. The WildReceipt and CORD checks above confirm that the gain is SROIE-specific.
- **Learns SROIE's label conventions and repeat vendors, not just reading.** The prompted baseline, the normalized scores and the seen/unseen split are included to separate these effects.
- **Label noise** in the gold data (see `B1750` above) caps strict scores.
- **Small test set:** 100 receipts, one seed and one training run. Read the McNemar p-values rather than small differences.
- **Overlap between splits:** the dataset card says the criterion for its train/test split is not known. Our key check rules out identical receipts, but 63 % of test receipts come from shops seen in training (measured above).
- **Known framework issues:**
  - Merged Gemma 4 weights produced garbled output (reported upstream), so these are adapters only.
  - Sampling hit NaN errors on some GPUs, so evaluation is greedy; batched generation can differ from batch size 1.
  - torch versions up to 2.13 have an SDPA prefill bug for prompts of length 1 mod 32 (reported on the 12B model). None of our 100 test prompts had such a length, for either instruction, and eager attention reproduced the SDPA outputs exactly on 3 + 3 spot-checked receipts.
- **Privacy:** receipts can contain names, card digits and phone numbers. Do not log or upload user receipts without consent.

## How to use

**Unsloth** (the route we used):

```python
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained("owaisnoe/gemma4-e2b-sroie-lora", load_in_4bit=False)  # base + adapters
FastVisionModel.for_inference(model)

INSTRUCTION = "Extract the company, date, address and total from this receipt. Reply with JSON only."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]  # IMAGE FIRST
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image.convert("RGB"), prompt, add_special_tokens=False, return_tensors="pt",
                   images_kwargs={"max_soft_tokens": 560}).to("cuda")                       # same 560 budget as training
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)                          # greedy, batch 1
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**PEFT + Transformers** (not tested by us):

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
base = AutoModelForImageTextToText.from_pretrained("unsloth/gemma-4-E2B-it", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "owaisnoe/gemma4-e2b-sroie-lora")
processor = AutoProcessor.from_pretrained("owaisnoe/gemma4-e2b-sroie-lora")
```

**Rules:**
- Keep the image **before** the text, and keep the **560** visual budget.
- Decode greedily at batch size 1.
- **Do not merge** the adapters into the base weights. Serve with Transformers or Unsloth, not vLLM.
- `model.disable_adapter()` gives you the plain model from the same loaded weights. The demo app uses this to show plain and fine-tuned side by side.

## Licence and attribution

- **Adapters:** Apache 2.0, the same as the Gemma 4 base model (the base model card links its licence to the [Gemma 4 licence page](https://ai.google.dev/gemma/docs/gemma_4_license), which shows Apache License 2.0).
- **Training data:** SROIE via `jsdnrs/ICDAR2019-SROIE`, CC BY 4.0 per the dataset card. The original terms are on the [RRC SROIE page](https://rrc.cvc.uab.es/?ch=13).
- **Evaluation-only data:** CORD v2 (`naver-clova-ix/cord-v2`, CC BY 4.0) and WildReceipt (`kaydee/wildreceipt`; Sun et al. 2021, arXiv:2103.14470; no licence stated on the Hub copy, used for evaluation only).

```bibtex
@inproceedings{Huang_2019,
  title     = {ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
  author    = {Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
  booktitle = {2019 International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2019},
  pages     = {1516--1520},
  doi       = {10.1109/ICDAR.2019.00244}
}
```
