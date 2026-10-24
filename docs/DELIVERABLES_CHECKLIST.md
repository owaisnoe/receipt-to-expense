<!--
PLACEHOLDERS: {{HF_REPO}} {{GRADIO_SHARE_URL}} {{RAG_ARTIFACT}} {{OOD_EXTERNAL_SET}} {{DRIVE_COPY_LINK}}
Status as of the docs fork write-up: notebook sections 0-5 done on the G4; plain baseline (section 6) running; prompted baseline (7) queued.
Owners use the parent's fork names. Confirm the assignments against the parent's plan.
-->

# Deliverables checklist (Theme 3B)

## Required: "at the showcase, the team must present"

| # | Deliverable | Evidence (notebook section / artifact) | Status | Owner |
|---|---|---|---|---|
| 1 | Fine-tuned model, with training settings documented | §8-11: LoRA, training, `gemma4_sroie_lora/` + `.zip`. §17: summary JSON. `docs/MODEL_CARD.md` settings table | in progress (training not started; card drafted) | main thread; docs |
| 2 | Dataset description: source, size, format, train/test separation | §3: 626/361 loaded, seed-0 100-receipt subset, `test_keys.json`, key-overlap assertion (ran OK). `docs/DATASET.md` | done (draft; fill `{{LOCAL_*}}`) | main thread; docs |
| 3 | Comparative analysis of plain vs fine-tuned on the same test set, with metrics **and example outputs** | §6 `eval_plain.csv`, §7 `eval_prompted.csv`, §12 `eval_finetuned.csv` + `comparison.csv` + `comparison.png` + McNemar; example outputs printed in §13 and stored in each CSV's `prediction` column | in progress (§6 running) | main thread |
| 4 | Working application around the fine-tuned model, demonstrated live | §15 Gradio app with `share=True`: fine-tuned beside plain, editable fields, expenses table, CSV. Link: {{GRADIO_SHARE_URL}} | planned | app (built), main thread (runs it in the kernel) |
| 5 | Discussion of where fine-tuning helped, where it did not, and why | §13 error categories, §14 local receipts, prompted-vs-fine-tuned McNemar; `docs/TALK_3MIN.md` 2:20-3:00; `docs/SLIDES_OUTLINE.md` slide 5 | planned (needs §12-14 numbers) | main thread; docs |

## Bonus points (from the brief's "Good to Have" list)

| Bonus | Evidence | Status | Owner |
|---|---|---|---|
| Prompted baseline | §7 `eval_prompted.csv`: same model, detailed rule prompt; McNemar fine-tuned vs prompted in §12 | in progress (queued after §6) | main thread |
| Training experiments (data size / steps / LoRA rank / LR) | Checkpoints saved every 40 steps (§10) → score checkpoints 40/80/120/157 on 50 receipts for a learning curve; optionally one rank or LR variant | planned | bg_runner (checkpoint curve); rag_experiments (variant), to be confirmed |
| Overfitting check (clearly different examples) | §14: local receipts + `labels.csv`, plain vs fine-tuned; optionally an external receipt set {{OOD_EXTERNAL_SET}} | planned (needs photos and labels) | ood_lineitems |
| Model size comparison (E4B: quality, speed, memory) | §16: E4B in a second process or runtime, with the same settings; all-four accuracy, s/receipt, peak GB | planned | bg_runner |
| Combined approach (RAG) | {{RAG_ARTIFACT}} | planned | rag_experiments |
| Publishing on HF with a model card | §11 `push_to_hub` (needs `HF_TOKEN` in Colab Secrets + repo name {{HF_REPO}}); `docs/MODEL_CARD.md` (data, intended use, limitations) | in progress (card drafted; push blocked on token and repo name) | main thread (push); docs (card) |
| Error analysis (categorise + propose fixes) | §13 table (correct / format only / missing / wrong value) + examples; label-noise case `B1750`/`81750` already found; fixes listed in the model card and on slide 5 | planned (runs after §12) | main thread; docs |
| 3B: line items (item, quantity, price) | Zero-shot extra on the fine-tuned or plain model, **clearly labelled as unevaluated** (SROIE has no line-item labels); qualitative examples only | planned | ood_lineitems |

## Presentation and safety items

| Item | Evidence | Status | Owner |
|---|---|---|---|
| 3-minute script + judge Q&A + demo backup plan | `docs/TALK_3MIN.md` | done (placeholders pending) | docs |
| Slides (6 maximum) | `docs/SLIDES_OUTLINE.md` | done (outline) | docs → team |
| Backup recording of the demo | screen recording during rehearsal | planned | team |
| Notebook saved to Drive | File > Save a copy in Drive → {{DRIVE_COPY_LINK}} | planned (end of run) | main thread |
| All evaluation CSVs kept | `/content/hackday/eval_*.csv`, `comparison.csv`, `test_keys.json`, `versions.txt` | in progress | main thread |
| Hard rules | No training on the test split or local receipts; same instruction, decoding, budget and load setting across the compared configurations | holding (assertion in §3; pixel parity verified in §4b) | main thread |
