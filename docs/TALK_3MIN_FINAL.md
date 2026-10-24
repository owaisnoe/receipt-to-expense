# 3-minute talk: Receipt to Expense (Theme 3B), final numbers

Spoken budget is about 140 words/min. Use one presenter and one demo driver. Rehearse twice with a timer, and record a 60-s backup video of the demo.
- App: https://f6753b53a276c78344.gradio.live (QR code on the demo slide)
- Model card: https://huggingface.co/owaisnoe/gemma4-e2b-sroie-lora

## 0:00-0:25 · Problem and data (about 55 words)
*(Hold up a real receipt.)*
> Riya's hostel flat splits groceries every week. The receipts become a WhatsApp photo pile, and someone types them into a spreadsheet.
> We built Receipt to Expense: one open Gemma 4 model, fine-tuned with LoRA, reads the photo, fills the expense, and splits the bill.
> Data: SROIE, 626 training receipts, and 100 fixed test receipts it never saw.

## 0:25-1:10 · One chart (about 100 words)
*(Slide: comparison chart, with the all-four bars highlighted.)*
> Same 100 receipts, same image budget, greedy decoding, three models.
> Plain Gemma gets all four fields exactly right on 6 percent. Given a detailed prompt with the label rules, 17. Our fine-tune: 63.
> Normalized scoring forgives formatting: 28, 37, 66.
> So it's not just formatting. Against the prompted model, McNemar gives p of one in a million.
> The biggest gain is the total: 73 to 95 percent.
> And what moved accuracy most wasn't model size. Doubling the visual tokens took us to 76; the twice-as-big E4B only reached 69.

## 1:10-2:20 · Live demo (about 100 words plus actions)
*(Open the app and photograph a judge's receipt.)*
> Could one of you lend us a receipt? Scan the QR to try it on your phone.
*(Extract: about 7 s.)*
> The fine-tuned model fills the form. Below it, the plain model's answer, field by field.
> The trust check says where both agree; on our test set, agreed fields were right 83 to 100 percent of the time, so it tells you what to double-check.
*(Scroll to Split the bill: names, Calculate split, Add to group balances.)*
> Shared it with friends? Split equally or by item. Tax is shared in proportion, and it settles up with the fewest payments.
> Add to expenses, download the CSV. Done.
**Skip line items live (about 9 s) unless asked.** If anything stalls for more than 15 s, play the backup video.

## 2:20-3:00 · Where it failed and what's next (about 85 words)
*(Slide: seen vs unseen shops, plus out-of-distribution results.)*
> Honest part. 63 of our test receipts came from shops seen in training. There it scores 83 percent; on new shops 38, still double the prompted model.
> On other countries' receipts the gain is flat, so it learned SROIE.
> We also caught a silent bug: the training pipeline shrank receipts to 512 pixels. We made training and test pixels identical.
> Next: Indian receipts, on-device 4-bit, UPI requests from the split.
> Adapters and model card are on Hugging Face.

## Judge questions (about 20 s each)
1. **"Isn't it just formatting?"** The prompted baseline gets the exact rules, and normalized scoring forgives format. Still 66 vs 37, p = 1e-6.
2. **"Test leakage?"** We use the separate test split, with keys asserted disjoint from training. The same shop can appear in both splits, so we measured it: 63%, with the seen/unseen split reported.
3. **"Why 560 tokens?"** Gemma 4 offers 70 to 1120. 560 keeps a typical receipt close to native size. We tested 280, 560 and 1120: 61, 66 and 76%, and 1120 vs 560 is significant (p = 0.006).
4. **"What was the bug?"** Same token count, different pixels: the collator shrank receipts to 512 px and snapped them to 32 px, with a pixel difference of up to 0.68. Passing the image as a column with `resize="max"` brought it to 0.0000.
5. **"What does LoRA change?"** The base weights are frozen. Rank-32 adapters on the vision and language layers mean 1.15% of the parameters, in a 240 MB file. `disable_adapter()` gives back the plain model, which is how the app shows both.
6. **"Paid compute?"** We trained on a Colab G4 for speed (2.5 min). Unsloth's docs put E2B LoRA at about 8-10 GB, so the recipe should also fit a free T4, more slowly. We did not test that.
