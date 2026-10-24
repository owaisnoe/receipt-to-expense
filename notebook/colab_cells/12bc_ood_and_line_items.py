# Colab cells for sections 12b (out-of-distribution check) and 12c (line items). Paste after section 12, in order.
# Needs the globals from sections 0-12: OUT, SEED, FIELDS, INSTRUCTION, model, FastVisionModel, generate_from_image,
# evaluate, plain_df, ft_df (and mcnemar, optional). Nothing here trains: the external sets are evaluation-only.

# %%
# CELL: 12b-1 OOD helpers: number-preserving JSON, amount/date/text normalizers, reading order, item matching (instant)
import os, re, io, json, time, random, difflib, contextlib, datetime as _dt

def parse_json_keep_numbers(text):
    """Like parse_json, but JSON numbers stay strings, so 60.000 (rupiah) is not turned into 60.0."""
    text = re.sub(r"```(?:json)?", "", str(text))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0), parse_float=str, parse_int=str)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None

def _flat(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_flat(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_flat(x) for x in v.values())
    return str(v).strip()

def _alnum(s):
    return re.sub(r"[^0-9A-Z]", "", _flat(s).upper())

def ood_amount(s, idr=False):
    """Canonical amount string. idr=True (CORD, Indonesian rupiah): '.' and ',' are thousands separators and only a
    trailing 1-2 digit group is cents (dropped): '60.000' -> '60000', '60.00' -> '60'. idr=False: a last separator
    followed by 1-2 digits is the decimal mark: 'Eur154,40' -> '154.40', '1,234.50' -> '1234.50'."""
    toks = re.findall(r"\d[\d.,]*", _flat(s))
    if not toks:
        return ""
    t = toks[-1].rstrip(".,")
    if idr:
        d = re.sub(r"\D", "", re.sub(r"[.,]\d{1,2}$", "", t))
        return str(int(d)) if d else ""
    m = re.match(r"^(.*?)[.,](\d{1,2})$", t)
    whole, frac = (m.group(1), m.group(2)) if m else (t, "0")
    whole = re.sub(r"\D", "", whole) or "0"
    return f"{float(whole + '.' + frac):.2f}"

_MON3 = {"JAN": 1, "ENE": 1, "FEB": 2, "FEV": 2, "FÉV": 2, "MAR": 3, "MÄR": 3, "MRZ": 3, "APR": 4, "ABR": 4, "AVR": 4,
         "MAY": 5, "MAI": 5, "MAG": 5, "JUN": 6, "JUL": 7, "AUG": 8, "AGO": 8, "AOU": 8, "AOÛ": 8, "SEP": 9, "SET": 9,
         "OCT": 10, "OKT": 10, "OTT": 10, "NOV": 11, "DEC": 12, "DEZ": 12, "DIC": 12, "DÉC": 12}

def _month(name):
    n = name.upper()
    return 6 if n.startswith("JUIN") else 7 if n.startswith("JUIL") else _MON3.get(n[:3])

def _iso(y, m, d):
    y = int(y)
    y = y + (2000 if y < 70 else 1900) if y < 100 else y
    try:
        return _dt.date(y, int(m), int(d)).isoformat() if 1990 <= y <= 2035 else None
    except ValueError:
        return None

def ood_dates(s):
    """Every ISO date a string can mean (day-first AND month-first), so US, EU and Asian receipts compare fairly."""
    s = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", _flat(s))          # 25Dec13 -> 25 Dec 13
    out = set()
    for a, b, c in re.findall(r"(?<!\d)(\d{1,4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})(?!\d)", s):
        cands = [(a, b, c)] if len(a) == 4 else [(c, b, a), (c, a, b)]             # (y, m, d)
        out |= {x for x in (_iso(*c3) for c3 in cands) if x}
    sep = r"\s*[\s\-/.,'’´`]?\s*"
    for d, mon, y in re.findall(r"(?<!\d)(\d{1,2})" + sep + r"([A-Za-zÀ-ÿ]{3,9})\.?" + sep + r"(\d{2,4})(?!\d)", s):
        if _month(mon):
            out |= {x for x in [_iso(y, _month(mon), d)] if x}
    for mon, d, y in re.findall(r"([A-Za-zÀ-ÿ]{3,9})\.?" + sep + r"(\d{1,2})\s*(?:st|nd|rd|th)?" + sep + r"(\d{2,4})(?!\d)", s):
        if _month(mon):
            out |= {x for x in [_iso(y, _month(mon), d)] if x}
    return out

def ood_text_match(pred, boxes, slack=2.5):
    """WildReceipt text has its spaces removed and is often only partly labelled: correct when every labelled box
    appears in the prediction (case, spaces, punctuation ignored) and the prediction is at most `slack` x longer."""
    bs = [b for b in dict.fromkeys(_alnum(x) for x in boxes) if b]
    if not bs:
        return None
    p = _alnum(pred)
    return int(bool(p) and all(b in p for b in bs) and len(p) <= slack * len("".join(bs)) + 2)

def ood_text_sim(pred, boxes):
    bs = [b for b in dict.fromkeys(_alnum(x) for x in boxes) if b]
    return round(difflib.SequenceMatcher(None, _alnum(pred), "".join(bs)).ratio(), 3) if bs else None

def ood_reading_order(idxs, bboxes):
    """Box indices in reading order: group boxes into lines by vertical centre, then left to right."""
    boxes = []
    for i in idxs:
        b = list(bboxes[i])
        if len(b) == 8:
            b = [min(b[0::2]), min(b[1::2]), max(b[0::2]), max(b[1::2])]
        boxes.append((i, b))
    if not boxes:
        return []
    hs = sorted(max(1, b[3] - b[1]) for _, b in boxes)
    tol, lines = 0.5 * hs[len(hs) // 2], []
    for i, b in sorted(boxes, key=lambda ib: (ib[1][1] + ib[1][3]) / 2):
        yc = (b[1] + b[3]) / 2
        if lines and abs(yc - lines[-1][0]) <= tol:
            lines[-1][1].append((b[0], i))
        else:
            lines.append([yc, [(b[0], i)]])
    return [i for _, ln in lines for _, i in sorted(ln)]

# WildReceipt class names, as in the dataset's ClassLabel feature (verified against the HF datasets-server /info)
WR_LABELS = ["Ignore", "Store_name_value", "Store_name_key", "Store_addr_value", "Store_addr_key", "Tel_value", "Tel_key",
             "Date_value", "Date_key", "Time_value", "Time_key", "Prod_item_value", "Prod_item_key", "Prod_quantity_value",
             "Prod_quantity_key", "Prod_price_value", "Prod_price_key", "Subtotal_value", "Subtotal_key", "Tax_value",
             "Tax_key", "Tips_value", "Tips_key", "Total_value", "Total_key", "Others"]
WR_FIELD_LABELS = {"company": "Store_name_value", "address": "Store_addr_value", "date": "Date_value", "total": "Total_value"}

def wildreceipt_gold(row):
    """{field: [box texts in reading order, duplicates dropped]} for the four SROIE fields."""
    tags = [WR_LABELS[t] if isinstance(t, int) else str(t) for t in row["ner_tags"]]
    gold = {}
    for field, label in WR_FIELD_LABELS.items():
        idxs = [i for i, t in enumerate(tags) if t == label]
        seen, vals = set(), []
        for i in ood_reading_order(idxs, row["bboxes"]):
            w = str(row["words"][i]).strip()
            if w and _alnum(w) not in seen:
                seen.add(_alnum(w)); vals.append(w)
        gold[field] = vals
    return gold

def cord_gold(gt_parse):
    """(items, total_price) from CORD's gt_parse. Items = menu lines plus priced sub-lines (toppings etc.)."""
    menu = gt_parse.get("menu") or []
    items = []
    for it in (menu if isinstance(menu, list) else [menu]):
        if not isinstance(it, dict):
            continue
        subs = it.get("sub") or []
        for x in [it] + [sb for sb in (subs if isinstance(subs, list) else [subs]) if isinstance(sb, dict)]:
            items.append({"name": _flat(x.get("nm")), "quantity": _flat(x.get("cnt")),
                          "price": _flat(x.get("price") or x.get("itemsubtotal")), "unitprice": _flat(x.get("unitprice"))})
    items = [x for x in items if ood_amount(x["price"], idr=True)]
    total = gt_parse.get("total") or {}
    if isinstance(total, list):
        total = next((t for t in total if isinstance(t, dict) and t.get("total_price")), {})
    return items, (_flat(total.get("total_price")) if isinstance(total, dict) else "")

def _name(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Z]+", " ", _flat(s).upper())).strip()

def _qty(s):
    m = re.search(r"\d+(?:[.,]\d+)?", _flat(s))
    if not m:
        return ""
    v = float(m.group(0).replace(",", "."))
    return str(int(v)) if v == int(v) else f"{v:g}"

def parse_line_items(raw):
    """(items, total, valid_json). Salvages complete item objects when the JSON was cut off by max_new_tokens."""
    obj = parse_json_keep_numbers(raw)
    if obj is not None and isinstance(obj.get("items"), list):
        src = obj["items"]
    else:
        src = []
        for frag in re.findall(r"\{[^{}]*\}", re.sub(r"```(?:json)?", "", str(raw))):
            try:
                o = json.loads(frag, parse_float=str, parse_int=str)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and {"name", "price"} & set(o):
                src.append(o)
    total = _flat((obj or {}).get("total"))
    if not total and obj is None:
        m = re.search(r'"total"\s*:\s*"?([^",}\n]+)', str(raw))
        total = m.group(1).strip() if m else ""
    items = [{"name": _flat(it.get("name")), "quantity": _flat(it.get("quantity")), "price": _flat(it.get("price"))}
             for it in src if isinstance(it, dict)]
    return items, total, obj is not None

def match_line_items(pred, gold, min_sim=0.8):
    """Greedy one-to-one matching: name similarity >= min_sim AND the same rupiah amount (line price or unit price)."""
    pairs = []
    for i, p in enumerate(pred):
        pp, pn = ood_amount(p.get("price"), idr=True), _name(p.get("name"))
        if not pp or not pn:
            continue
        for j, g in enumerate(gold):
            if pp in {ood_amount(g.get("price"), idr=True), ood_amount(g.get("unitprice"), idr=True)} - {""}:
                sim = difflib.SequenceMatcher(None, pn, _name(g.get("name"))).ratio()
                if sim >= min_sim:
                    pairs.append((sim, i, j))
    used_p, used_g, out = set(), set(), []
    for sim, i, j in sorted(pairs, key=lambda x: (-x[0], x[1], x[2])):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); out.append((i, j))
    return out

# %%
# CELL: 12b-2 Download only the CORD v2 and WildReceipt TEST files and build seeded subsets (~1-2 min, ~660 MB)
import pandas as pd
import datasets as hfds
from huggingface_hub import hf_hub_download
N_OOD = 50                       # receipts per external set, seeded; 30 cuts the OOD + line-item time by ~40%
OOD_DIR = f"{OUT}/ood"; os.makedirs(OOD_DIR, exist_ok=True)
OOD_SOURCES = {                  # (repo, test file, pinned revision); evaluation only, never trained on
    "cord": ("naver-clova-ix/cord-v2", "data/test-00000-of-00001-9c204eb3f4e11791.parquet", "7f0115a4b758a71d6473b8d085751692da2fef98"),
    "wildreceipt": ("kaydee/wildreceipt", "data/test-00000-of-00001.parquet", "cedafaf3c8b0246c9fad68af29324d655715ad12"),
}

def _load_test_parquet(name):
    repo, fname, rev = OOD_SOURCES[name]
    ds = hfds.Dataset.from_parquet(hf_hub_download(repo, fname, repo_type="dataset", revision=rev))
    if not isinstance(ds.features["image"], hfds.Image):        # CORD's parquet carries no HF feature metadata
        ds = ds.cast_column("image", hfds.Image())
    return ds

def _subset(n_total):
    return sorted(random.Random(SEED).sample(range(n_total), min(N_OOD, n_total)))

cord_ds = _load_test_parquet("cord")
cord_samples = []
for _i in _subset(len(cord_ds)):
    _r = cord_ds[_i]
    _items, _total = cord_gold(json.loads(_r["ground_truth"])["gt_parse"])
    cord_samples.append({"key": f"cord_test_{_i}", "image": _r["image"], "gold_items": _items,
                         "entities": {"company": "", "date": "", "address": "", "total": _total}})

wr_ds = _load_test_parquet("wildreceipt")
_names = getattr(getattr(wr_ds.features["ner_tags"], "feature", None), "names", None)
assert _names is None or list(_names) == WR_LABELS, f"WildReceipt label names differ: {_names}"
wr_samples = []
for _i in _subset(len(wr_ds)):
    _r = wr_ds[_i]
    _g = wildreceipt_gold(_r)
    wr_samples.append({"key": f"wr_test_{_r['id']}", "image": _r["image"], "gold_boxes": _g,
                       "entities": {k: " ".join(v) for k, v in _g.items()}})

json.dump({"cord": [s["key"] for s in cord_samples], "wildreceipt": [s["key"] for s in wr_samples]},
          open(f"{OOD_DIR}/ood_keys.json", "w"))
print(f"CORD v2 test: {len(cord_ds)} receipts -> {len(cord_samples)} (seed {SEED}); labelled: total "
      f"{sum(bool(s['entities']['total']) for s in cord_samples)}, line items {sum(len(s['gold_items']) for s in cord_samples)}")
print(f"WildReceipt test: {len(wr_ds)} receipts -> {len(wr_samples)} (seed {SEED}); labelled per field:",
      {k: sum(bool(s["gold_boxes"][k]) for s in wr_samples) for k in FIELDS})
print("example WildReceipt gold:", wr_samples[0]["entities"])
print("example CORD gold:", cord_samples[0]["entities"]["total"], cord_samples[0]["gold_items"][:3])

# %%
# CELL: 12b-3 WildReceipt (US/EU/Asia photos): plain vs fine-tuned, same INSTRUCTION (~5 min for 2 x 50 receipts)
def _run_quiet(samples, label, plain):
    """evaluate() from section 5: same INSTRUCTION, greedy, batch 1, same visual budget. Its SROIE-style printout is
    hidden (SROIE's normalizers do not fit these sets); its raw-output CSV is moved into OOD_DIR."""
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = evaluate(samples, label, plain=plain)
    if os.path.exists(f"{OUT}/eval_{label}.csv"):
        os.replace(f"{OUT}/eval_{label}.csv", f"{OOD_DIR}/eval_{label}.csv")
    dt = time.time() - t0
    print(f"{label}: {len(samples)} receipts in {dt / 60:.1f} min ({dt / max(1, len(samples)):.1f} s/receipt)")
    return df

def score_wildreceipt(df, samples):
    gold = {s["key"]: s["gold_boxes"] for s in samples}
    rows = []
    for key, raw in zip(df["key"], df["prediction"]):
        g, p = gold[key], parse_json_keep_numbers(raw)
        row = {"key": key, "prediction": raw, "valid_json": int(p is not None)}
        p = p or {}
        for k in ("company", "address"):
            row[f"{k}_norm"], row[f"{k}_sim"] = ood_text_match(p.get(k), g[k]), ood_text_sim(p.get(k), g[k])
        gd = set().union(*[ood_dates(b) for b in g["date"]])
        row["date_norm"] = int(bool(ood_dates(p.get("date")) & gd)) if gd else None
        gt = {ood_amount(b) for b in g["total"]} - {""}
        row["total_norm"] = int(ood_amount(p.get("total")) in gt) if gt else None
        lab = [row[f"{k}_norm"] for k in FIELDS if row[f"{k}_norm"] is not None]
        row["all_labelled_norm"] = int(all(lab)) if lab else None
        rows.append(row)
    return pd.DataFrame(rows)

def _mcnemar_pair(a, b):
    m = a.notna() & b.notna()
    x, y = a[m].astype(int).values, b[m].astype(int).values
    if "mcnemar" in globals():
        return mcnemar(x, y)
    from scipy.stats import binomtest
    better, worse = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    return better, worse, (binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0)

def side_by_side(plain, ft, rows):
    """rows: [(label, column)]; only receipts where the field is labelled count; McNemar for 0/1 columns."""
    out = []
    for label, col in rows:
        a, b = plain[col].astype(float), ft[col].astype(float)
        m = a.notna() & b.notna()
        r = {"metric": label, "n labelled": int(m.sum()), "Plain (%)": round(100 * a[m].mean(), 1),
             "Fine-tuned (%)": round(100 * b[m].mean(), 1)}
        if not col.endswith("_sim"):
            bb, ww, pv = _mcnemar_pair(a, b)
            r.update({"improved/worse": f"{bb}/{ww}", "McNemar p": float(f"{pv:.2g}")})
        out.append(r)
    return pd.DataFrame(out).set_index("metric")

wr_plain = score_wildreceipt(_run_quiet(wr_samples, "ood_wildreceipt_plain", plain=True), wr_samples)
wr_ft = score_wildreceipt(_run_quiet(wr_samples, "ood_wildreceipt_finetuned", plain=False), wr_samples)
wr_plain.to_csv(f"{OOD_DIR}/wildreceipt_plain_scored.csv", index=False)
wr_ft.to_csv(f"{OOD_DIR}/wildreceipt_finetuned_scored.csv", index=False)
WR_TABLE = side_by_side(wr_plain, wr_ft, [
    ("valid JSON", "valid_json"), ("company", "company_norm"), ("company similarity", "company_sim"),
    ("address", "address_norm"), ("address similarity", "address_sim"), ("date (either day/month order)", "date_norm"),
    ("total", "total_norm"), ("all labelled fields (receipt)", "all_labelled_norm")])
WR_TABLE.to_csv(f"{OOD_DIR}/wildreceipt_table.csv")
print(f"\nWildReceipt test subset ({len(wr_samples)} receipts, seed {SEED}), labelled fields only:\n" + WR_TABLE.to_string())

# %%
# CELL: 12b-4 CORD v2 (Indonesian receipts, only the total is labelled): plain vs fine-tuned (~4-5 min for 2 x 50)
def score_cord_total(df, samples):
    gold = {s["key"]: s["entities"]["total"] for s in samples}
    rows = []
    for key, raw in zip(df["key"], df["prediction"]):
        p = parse_json_keep_numbers(raw)
        g, pv = ood_amount(gold[key], idr=True), ood_amount((p or {}).get("total"), idr=True)
        err = ("unlabelled" if not g else "correct" if pv == g else "missing" if not pv else
               "thousands misread (x1000)" if int(pv) * 1000 == int(g) else "wrong value")
        rows.append({"key": key, "prediction": raw, "valid_json": int(p is not None), "gold_total": gold[key],
                     "pred_total": _flat((p or {}).get("total")), "total_norm": int(pv == g) if g else None, "total_error": err})
    return pd.DataFrame(rows)

cord_plain = score_cord_total(_run_quiet(cord_samples, "ood_cord_plain", plain=True), cord_samples)
cord_ft = score_cord_total(_run_quiet(cord_samples, "ood_cord_finetuned", plain=False), cord_samples)
cord_plain.to_csv(f"{OOD_DIR}/cord_plain_scored.csv", index=False)
cord_ft.to_csv(f"{OOD_DIR}/cord_finetuned_scored.csv", index=False)
CORD_TABLE = side_by_side(cord_plain, cord_ft, [("valid JSON", "valid_json"), ("total (rupiah, separators ignored)", "total_norm")])
CORD_TABLE.to_csv(f"{OOD_DIR}/cord_table.csv")
print(f"\nCORD v2 test subset ({len(cord_samples)} receipts, seed {SEED}):\n" + CORD_TABLE.to_string())
print("\nTotal errors:\n" + pd.DataFrame({"Plain": cord_plain["total_error"].value_counts(),
                                           "Fine-tuned": cord_ft["total_error"].value_counts()}).fillna(0).astype(int).to_string())

# %%
# CELL: 12b-5 Overfitting check: in-domain (SROIE) vs out-of-domain change, McNemar, chart (instant)
import matplotlib.pyplot as plt
_gen = []
def _gen_row(dataset, metric, a, b):
    t = side_by_side(pd.DataFrame({"m": a.values}), pd.DataFrame({"m": b.values}), [(metric, "m")]).reset_index()
    t.insert(0, "dataset", dataset)
    _gen.append(t)
if "plain_df" in globals() and "ft_df" in globals():
    _gen_row("SROIE (in-domain)", "all 4 fields", plain_df["all4_norm"], ft_df["all4_norm"])
    _gen_row("SROIE (in-domain)", "total", plain_df["total_norm"], ft_df["total_norm"])
_gen_row("WildReceipt (OOD)", "all labelled fields", wr_plain["all_labelled_norm"], wr_ft["all_labelled_norm"])
for _k in FIELDS:
    _gen_row("WildReceipt (OOD)", _k, wr_plain[f"{_k}_norm"], wr_ft[f"{_k}_norm"])
_gen_row("CORD (OOD)", "total", cord_plain["total_norm"], cord_ft["total_norm"])
GENERALISATION = pd.concat(_gen, ignore_index=True)
GENERALISATION["change (pts)"] = (GENERALISATION["Fine-tuned (%)"] - GENERALISATION["Plain (%)"]).round(1)
GENERALISATION.to_csv(f"{OOD_DIR}/generalisation.csv", index=False)
print(GENERALISATION.to_string(index=False))
_chg = GENERALISATION.set_index(["dataset", "metric"])["change (pts)"]
print("\nTotal-field change from fine-tuning: " + ", ".join(f"{d.split(' (')[0]} {_chg[(d, 'total')]:+.1f} pts"
      for d in GENERALISATION["dataset"].unique() if (d, "total") in _chg.index))
_ax = GENERALISATION.set_index(GENERALISATION["dataset"].str.split(" ").str[0] + ": " + GENERALISATION["metric"])[
    ["Plain (%)", "Fine-tuned (%)"]].plot(kind="barh", figsize=(8, 0.45 * len(GENERALISATION) + 1.2))
_ax.invert_yaxis(); _ax.set_xlabel("% correct (labelled fields only)")
_ax.set_title("Does the SROIE fine-tune generalise? Plain vs fine-tuned Gemma 4 E2B")
plt.tight_layout(); plt.savefig(f"{OOD_DIR}/generalisation.png", dpi=150); plt.show()

# %%
# CELL: 12c-1 Line items (bonus): prompt and extract_line_items(), plain model by default (instant)
LINE_ITEMS_PROMPT = (
    "Extract every purchased line item and the total from this receipt. Reply with one JSON object only, in this format: "
    '{"items": [{"name": "...", "quantity": "...", "price": "..."}], "total": "..."}\n'
    "Rules:\n"
    "- name: the item name exactly as printed.\n"
    "- quantity: the quantity as printed; use \"1\" if none is printed.\n"
    "- price: the amount printed on that item's line (quantity x unit price), without a currency symbol.\n"
    "- total: the final amount paid, as printed, without a currency symbol.\n"
    "List each item line once, top to bottom. No explanation, no markdown."
)
LINE_ITEMS_MAX_NEW_TOKENS = 512

def extract_line_items(image, use_adapter=False, max_new_tokens=LINE_ITEMS_MAX_NEW_TOKENS):
    """Returns (items, raw, seconds); items = [{"name", "quantity", "price"}], total = parse_line_items(raw)[1].
    Plain model by default: the adapter was trained on the four fields only. Same visual budget, greedy, batch 1.
    The model must be in inference mode (FastVisionModel.for_inference(model), done in sections 4, 5 and 15)."""
    t0 = time.time()
    if use_adapter or not hasattr(model, "disable_adapter"):
        raw = generate_from_image(image, LINE_ITEMS_PROMPT, max_new_tokens)
    else:
        with model.disable_adapter():
            raw = generate_from_image(image, LINE_ITEMS_PROMPT, max_new_tokens)
    return parse_line_items(raw)[0], raw, time.time() - t0

# %%
# CELL: 12c-2 Line items on the CORD subset: plain vs fine-tuned (~6-10 min for 2 x 50; outputs ~100-300 tokens)
def eval_line_items(samples, label, use_adapter):
    FastVisionModel.for_inference(model)
    rows, t0 = [], time.time()
    for n, s in enumerate(samples, 1):
        items, raw, secs = extract_line_items(s["image"], use_adapter=use_adapter)
        _, total, valid = parse_line_items(raw)
        gold, gtotal = s["gold_items"], ood_amount(s["entities"]["total"], idr=True)
        pairs = match_line_items(items, gold)
        qty = [(_qty(items[i]["quantity"]), _qty(gold[j]["quantity"])) for i, j in pairs if _qty(gold[j]["quantity"])]
        pp, gp, pt = [ood_amount(x["price"], idr=True) for x in items], [ood_amount(x["price"], idr=True) for x in gold], ood_amount(total, idr=True)
        rows.append({"key": s["key"], "raw": raw, "pred_items": json.dumps(items, ensure_ascii=False),
                     "gold_items": json.dumps(gold, ensure_ascii=False), "valid_json": int(valid),
                     "n_pred": len(items), "n_gold": len(gold), "tp": len(pairs),
                     "perfect": int(len(pairs) == len(gold) == len(items)),
                     "qty_ok": sum(a == b for a, b in qty), "qty_n": len(qty),
                     "total_ok": int(pt == gtotal) if gtotal else None,
                     "pred_sum_eq_total": int(bool(items) and bool(pt) and all(pp) and sum(map(int, pp)) == int(pt)),
                     "gold_sum_eq_total": int(sum(map(int, gp)) == int(gtotal)) if gold and gtotal else None,
                     "seconds": round(secs, 2)})
        if n % 10 == 0 or n == len(samples):
            print(f"\r{label}: {n}/{len(samples)} ({(time.time() - t0) / n:.1f} s/receipt)", end="")
    print()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OOD_DIR}/lineitems_cord_{label}.csv", index=False)
    return df

li_plain = eval_line_items(cord_samples, "plain", use_adapter=False)
li_ft = eval_line_items(cord_samples, "finetuned", use_adapter=True)

# %%
# CELL: 12c-3 Line-item results: item precision/recall/F1, totals, sum check, McNemar, two examples (instant)
def li_summary(df):
    tp, npred, ngold = df["tp"].sum(), df["n_pred"].sum(), df["n_gold"].sum()
    P, R = (tp / npred if npred else 0.0), (tp / ngold if ngold else 0.0)
    return {"valid JSON (%)": 100 * df["valid_json"].mean(), "item precision (%)": 100 * P, "item recall (%)": 100 * R,
            "item F1 (%)": 100 * (2 * P * R / (P + R) if P + R else 0.0), "receipts with every item right (%)": 100 * df["perfect"].mean(),
            "quantity right, matched items (%)": 100 * df["qty_ok"].sum() / max(1, df["qty_n"].sum()),
            "total right (%)": 100 * df["total_ok"].astype(float).mean(),
            "sum(items) = total, model output (%)": 100 * df["pred_sum_eq_total"].mean(),
            "sum(items) = total, gold labels (%)": 100 * df["gold_sum_eq_total"].astype(float).mean(),
            "s/receipt": df["seconds"].mean()}
LINE_ITEMS_TABLE = pd.DataFrame({"Plain": li_summary(li_plain), "Fine-tuned": li_summary(li_ft)}).round(1)
LINE_ITEMS_TABLE.to_csv(f"{OOD_DIR}/lineitems_table.csv")
print(f"Line items on {len(cord_samples)} CORD test receipts ({int(li_plain['n_gold'].sum())} gold item lines); "
      "match = name similarity >= 0.8 AND same amount:\n" + LINE_ITEMS_TABLE.to_string())
_b, _w, _p = _mcnemar_pair(li_plain["perfect"], li_ft["perfect"])
print(f"\nFine-tuned vs plain, receipts with every item right: {_b} improved, {_w} got worse, exact McNemar p = {_p:.2g}")
for _j in li_plain.sort_values("n_gold", ascending=False).index[:2]:
    print(f"\n[{li_plain.at[_j, 'key']}] gold: " + "; ".join(f"{x['name']} x{x['quantity'] or 1} = {x['price']}"
                                                   for x in json.loads(li_plain.at[_j, "gold_items"])))
    for _lab, _df in (("plain", li_plain), ("fine-tuned", li_ft)):
        print(f"  {_lab}: " + ("; ".join(f"{x['name']} x{x['quantity']} = {x['price']}" for x in json.loads(_df.at[_j, "pred_items"]))
                               or "(no items) raw: " + _df.at[_j, "raw"][:150].replace("\n", " ")))
