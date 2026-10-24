<!--
PLACEHOLDERS: {{LOCAL_N}} {{LOCAL_SOURCE}} {{LOCAL_LABELLER}} {{VERIFY_B1750_IMAGE}} {{OOD_EXTERNAL_SET}}
-->

# Dataset description (Theme 3B)

## Source and licence

**Origin.** SROIE (Scanned Receipt OCR and Information Extraction), from the ICDAR 2019 Robust Reading Competition.
- Paper: Huang et al., 2019, [arXiv:2103.10213](https://arxiv.org/abs/2103.10213), DOI 10.1109/ICDAR.2019.00244.
- The paper describes 1,000 scanned receipt images split into 600 "trainval" and 400 "test".
- Our task is **Task 3, "Key Information Extraction from Scanned Receipts"**. Its gold JSON has four fields: `company`, `date`, `address`, `total`.
- The paper notes that some sensitive fields (name, address, contact number) were blurred on some receipts for privacy.

**Copy we used.** [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) on Hugging Face.
- **Licence:** CC BY 4.0 according to the card. The original terms are on the [RRC SROIE page](https://rrc.cvc.uab.es/?ch=13). We did not check those terms, and other mirrors list different licences.
- **Contents:** the card says this copy adds "14 receipt annotations missing from the original Task 3 test dataset" and fixes some errors in the source data and image metadata.
- **Splits:** 987 receipts, with **626 train / 361 test**. We confirmed both counts when loading it in the notebook.
- **Split criterion:** the card states that "the criterion for splitting the data is not known". Its prose also says 347 test images; 361 - 347 = 14, which matches the added annotations.

## Format

The table shows one row per receipt (column types as listed by the Hugging Face viewer).

| Column | Type | Used by us |
|---|---|---|
| `image` | image (width 436 to about 4,960 px; our largest training receipt is 4961x7016) | yes, as model input |
| `key` | 12-character id, e.g. `X00016469612` | yes, for split bookkeeping |
| `entities` | dict: `company`, `date`, `address`, `total` (strings) | yes, as the target / gold |
| `image_size` | dict: width, height | no |
| `words`, `bboxes` | OCR text segments and 4-integer boxes (18-153 per receipt) | **no**: the model reads pixels end to end, with no OCR input |

**Annotation conventions.** These matter for strict scoring.
- **Dates:** kept **exactly as printed**, and formats vary. Examples: `25/12/2018` and `15/01/2019`. Our earlier inspection also found `12-01-19`, `18-11-18` and `23-01-2019`.
- **Totals:** bare numbers with no currency, e.g. `9.00` and `193.00`. A plain model writing `193.00 SR` misses on strict scoring but scores on normalized.
- **Company and address:** printed text, mostly in capitals, punctuation as printed.
- **Example training target:**
  `{"company": "BOOK TA .K (TAMAN DAYA) SDN BHD", "date": "25/12/2018", "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.", "total": "9.00"}`
- **Label noise:** in one test receipt, the gold address contains `B1750 MASAI, JOHOR`, while the model read `81750`. Malaysian postcodes are five digits, so this is most likely an OCR slip in the gold label ({{VERIFY_B1750_IMAGE}}). Labels were not edited.
- **Geography:** we observed Malaysian receipts (Johor and Selangor addresses, RM amounts, GST lines). The sources we checked do not state this.

## How train and test were separated

1. **Training:** all 626 receipts of the `train` split. Each becomes a chat with the image first, then the instruction, and the gold JSON as the assistant answer.
2. **Test:**
   - 100 receipts drawn **once** from the 361-receipt `test` split, using `sorted(random.Random(0).sample(range(361), 100))`.
   - Their keys are saved to `test_keys.json`.
   - Every configuration (plain, prompted, fine-tuned) is scored on exactly these 100.
3. **Leakage guard:** the notebook asserts that no key in the test subset appears in the training split (`assert not train_keys & test_keys`).
4. **Never trained on:**
   - any receipt from the `test` split;
   - the team's local receipts;
   - any external receipt set used for robustness ({{OOD_EXTERNAL_SET}}).
5. **Caveat:** the key check rules out identical receipts, not the same shop appearing in both splits. The split criterion is unknown, so company and address scores may benefit from repeated shops. The local-receipt check addresses this.

## Local / out-of-distribution receipts (robustness only)

- **What:** {{LOCAL_N}} receipts photographed by the team ({{LOCAL_SOURCE}}), labelled by hand ({{LOCAL_LABELLER}}) in `labels.csv` with the columns `file,company,date,address,total`.
- **Use:** evaluation only (notebook section 14), scored with the same strict and normalized rules.
- **Privacy:** crop or blur card numbers, names and phone numbers. Do not publish the photos.
