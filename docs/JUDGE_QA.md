# Expected judge questions: Receipt to Expense (Theme 3B)

Answer each in about 20 seconds. Every number here is measured and public in the HF model card, so quote it confidently.
"Best version" means E2B at 1120 visual tokens; all tests use 100 held-out SROIE receipts.

## Results
1. **"How much better than the plain model, really?"**
   Same base model, same settings, with vs without our fine-tune:
   - receipts read perfectly: 7% → 70%, so **10×**;
   - field errors cut by **three quarters**: field accuracy 70% → 92%.
2. **"Isn't that just formatting? A good prompt would do it."**
   We tested exactly that. A prompt that spells out the label rules gets 37% of receipts fully right; ours gets **76%, 2×**. Our scoring also forgives case, spacing, currency and date format, so this measures reading, not formatting.
3. **"Is 100 receipts enough to trust that?"**
   It's a paired test on the same receipts. Against the engineered prompt, 40 receipts flipped to correct and 1 flipped the other way (McNemar **p < 1e-10**). That's far beyond noise.
4. **"Did you just pick your best run on the test set?"**
   Partly, and we say so. The 1120 version was the best of 5 configurations. But our *pre-planned* first model already beat the prompt 66% vs 37% (p = 1e-6), so the edge doesn't depend on that pick. Next step: a fresh test set.
5. **"What's the accuracy per field?"**
   Total 98%, date 98%, merchant 93%, address 80%. That's 92% of all fields.

## Model and method
6. **"Why Gemma 4 E2B?"** It's an open, 2B-class multimodal model built for phones and laptops. It reads the photo directly, with no OCR step, and fine-tunes in minutes on one GPU.
7. **"What does LoRA change?"** The base weights stay frozen. We train small rank-32 adapters on the vision and language layers: 1.15% of the parameters, in a 240 MB file. Switch them off and you get the original model back; the app does this to show both side by side.
8. **"Why did resolution beat model size?"** Receipts are small print. Going to 1120 visual tokens (near native resolution) took us from 66% to 76%. The twice-as-large E4B only reached 69%. Reading needs pixels more than parameters.
9. **"What bug did you find?"** The training pipeline silently shrank receipts to 512 px, while the model saw full-size images at test time. The token counts matched, so normal checks missed it. We compared the pixels directly, found differences up to 0.68, and fixed them to exactly 0.
10. **"How much data and time?"** 626 receipts, one epoch: 2.5 minutes of training for the first model. Most of the gain arrives after about 160 receipts.

## Data and leakage
11. **"Any test leakage?"** No. We use the official separate test split, the code asserts that test keys never appear in training, and we never trained on test, external or local receipts.
12. **"Do the same shops appear in train and test?"** Yes, and we measured it: 63% of test receipts are from shops seen in training. On brand-new shops the model still roughly **doubles** the prompted model.
13. **"Will it work on Indian receipts?"** On two other receipt datasets (US/EU and Indonesian) it does not break: there's no significant loss, and totals even improve. Indian receipts are the next fine-tune, about an afternoon's work once we have a labelled set.

## App and product
14. **"How is this different from Expensify, Google Lens, or OCR + ChatGPT?"**
    - There's no OCR pipeline, no cloud API and no subscription: it's one open model you can self-host.
    - It tells you which field to double-check.
    - It's built for students: split the bill and settle up with the fewest payments.
15. **"How does the split handle tax and service?"** Each item is split among the people who shared it. Tax, service and rounding are split in proportion to each person's items, and the leftover cent goes to whoever paid. Group balances settle in the fewest payments.
16. **"Privacy?"** Photos are not stored and not used for training. The model is small and open, and the next step is a 4-bit on-device version so receipts never leave the phone.
17. **"How fast?"** A few seconds of model time per receipt on a GPU. The rest is network.
18. **"What if it reads something wrong?"** Every field is editable. The app flags fields to double-check and says "please check" when the date or total doesn't parse: a human stays in the loop.

## Limits and next steps
19. **"Where does it fail?"** Long addresses with one- or two-character slips. Some of these are actually typos in the dataset's own labels: one label says `B1750` where the receipt prints `81750`. It also struggles with handwriting and non-Latin scripts.
20. **"What would you do with more time?"** Indian receipts and our own photos, a 4-bit on-device version, UPI payment requests straight from the settle-up list, and fine-tuning for line items.
21. **"Did you use paid compute?"** We trained on a Colab G4 for speed. The recipe needs about 8-10 GB, so it should also run on a free T4, more slowly; we didn't test that.
22. **"Line items?"** These are zero-shot, because the dataset has no item labels. Item F1 is 79% on Indonesian receipts, and our fine-tune didn't hurt it.
23. **"Who did what?"** *(Each teammate: one sentence.)*

## If you get a curveball
Say what we measured, then what we'd measure next. "We didn't test that; here's how we would" scores better than guessing.
