# Colab cells: RAG vendor memory (bonus "Combined approach"), checkpoint learning curve (bonus "Training experiments")
# and aggregation of the background experiments. Paste each cell block below as its own cell AFTER section 12.
# Order: RAG (CPU only) -> learning curve (GPU; run it BEFORE the Gradio app is live) -> aggregation.

# %%
# CELL: RAG bonus - vendor memory from TRAIN labels, applied to the stored predictions (about 30-60 s, CPU only, no generation)
# Retrieval-augmented post-correction. The predicted (company, address) is matched against a "vendor memory" built
# ONLY from the 626 TRAIN labels. When the nearest train vendor is at least RAG_TAU similar on BOTH fields, its train
# spelling replaces the prediction. Nothing is regenerated, so the instruction, greedy batch-1 decoding, the 560
# visual budget and the load setting are exactly those of the stored runs.
import os, re, json, math, difflib
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

RAG_DIR = f"{OUT}/rag"; os.makedirs(RAG_DIR, exist_ok=True)
RAG_TAU = 0.85     # fixed a priori, before any RAG result was seen (the sweep at the end is post-hoc)
RAG_TOPK = 10      # candidates from the character-trigram prefilter, re-ranked with difflib

def _mkey(s):
    """Similarity key only: upper case, punctuation -> space, single spaces. Replacements use the raw train strings."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Z]+", " ", str(s or "").upper())).strip()

def _grams(s, n=3):
    s = f" {s} "
    return Counter(s[i:i + n] for i in range(max(1, len(s) - n + 1)))

def _gnorm(g):
    return math.sqrt(sum(v * v for v in g.values()))

def _cos(ga, na, gb, nb):
    if not na or not nb:
        return 0.0
    if len(ga) > len(gb):
        ga, gb = gb, ga
    return sum(v * gb.get(k, 0) for k, v in ga.items()) / (na * nb)

def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def _mcnemar_exact(base, new):
    base, new = np.asarray(base), np.asarray(new)
    b, w = int(((base == 0) & (new == 1)).sum()), int(((base == 1) & (new == 0)).sum())
    n, k = b + w, min(b, w)
    return b, w, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))
_mc = globals().get("mcnemar", _mcnemar_exact)   # section 12's exact McNemar if it ran

# 1. Vendor memory from TRAIN labels only (the image column is never decoded here)
try:
    _train_label_rows = list(train_data.select_columns(["key", "entities"]))
except Exception:
    _train_label_rows = [{"key": train_data[i]["key"], "entities": train_data[i]["entities"]} for i in range(len(train_data))]
_test_keys = {s["key"] for s in test_subset}
_by_vendor = defaultdict(lambda: {"company": Counter(), "address": Counter(), "keys": []})
for _r in _train_label_rows:
    _c = str(_r["entities"].get("company") or "").strip()
    _a = str(_r["entities"].get("address") or "").strip()
    if _mkey(_c) and _mkey(_a):
        _v = _by_vendor[(_mkey(_c), _mkey(_a))]
        _v["company"][_c] += 1; _v["address"][_a] += 1; _v["keys"].append(_r["key"])
_index_keys = {k for v in _by_vendor.values() for k in v["keys"]}
assert _index_keys <= {r["key"] for r in _train_label_rows}, "vendor memory must come from the train split only"
assert _index_keys.isdisjoint(_test_keys), "a test receipt's label went into the vendor memory"
if "test_data" in globals():
    _shared = _index_keys & set(test_data["key"])
    print(f"train keys that also occur in the full test split: {len(_shared)}" + (" (CHECK THIS)" if _shared else ""))

VENDOR_MEMORY = []
for (_mc_key, _ma_key), _v in _by_vendor.items():
    _gc, _ga = _grams(_mc_key), _grams(_ma_key)
    VENDOR_MEMORY.append({"company": _v["company"].most_common(1)[0][0], "address": _v["address"].most_common(1)[0][0],
                          "company_variants": set(_v["company"]), "address_variants": set(_v["address"]),
                          "mc": _mc_key, "ma": _ma_key, "gc": _gc, "ga": _ga, "nc": _gnorm(_gc), "na": _gnorm(_ga),
                          "dc": re.findall(r"\d+", _mc_key), "da": re.findall(r"\d+", _ma_key),
                          "n_train_receipts": len(_v["keys"])})
pd.DataFrame([{k: v[k] for k in ("company", "address", "n_train_receipts")} for v in VENDOR_MEMORY]) \
    .sort_values("n_train_receipts", ascending=False).to_csv(f"{RAG_DIR}/vendor_memory.csv", index=False)
print(f"vendor memory: {len(VENDOR_MEMORY)} distinct (company, address) pairs from {len(_index_keys)} TRAIN receipts; "
      f"no test receipt used (asserted)")

_near_cache = {}
def _nearest(company, address):
    """(similarity, sim_company, sim_address, vendor, numbers_match) for the closest TRAIN vendor, or None if a field
    is empty. similarity = min(company ratio, address ratio): both fields must agree with the same vendor.
    numbers_match: the digit runs (lot numbers, postcode, "99" in a name) of each predicted field equal the vendor's.
    Text similarity alone cannot tell branch 3 from branch 31, so a vendor whose numbers differ is never a match."""
    qc, qa = _mkey(company), _mkey(address)
    if not qc or not qa or not VENDOR_MEMORY:
        return None
    if (qc, qa) not in _near_cache:
        gqc, gqa = _grams(qc), _grams(qa); nqc, nqa = _gnorm(gqc), _gnorm(gqa)
        dqc, dqa = re.findall(r"\d+", qc), re.findall(r"\d+", qa)
        pre = sorted(VENDOR_MEMORY, key=lambda v: -(_cos(gqc, nqc, v["gc"], v["nc"]) + _cos(gqa, nqa, v["ga"], v["na"])))
        best = None
        for v in pre[:RAG_TOPK]:
            sc, sa = _ratio(qc, v["mc"]), _ratio(qa, v["ma"])
            digits = (not dqc or dqc == v["dc"]) and (not dqa or dqa == v["da"])
            rank = (digits, min(sc, sa), sc + sa, v["n_train_receipts"])
            if best is None or rank > best[0]:
                best = (rank, sc, sa, v)
        _near_cache[(qc, qa)] = (best[0][1], best[1], best[2], best[3], best[0][0])
    return _near_cache[(qc, qa)]

def _is_match(r, tau=None):
    return r is not None and r[4] and r[0] >= (RAG_TAU if tau is None else tau)

def vendor_nearest(company, address):
    """Closest TRAIN vendor whatever its similarity (for display), or None if company or address is empty."""
    r = _nearest(company, address)
    if r is None:
        return None
    sim, sc, sa, v, digits = r
    return {"company": v["company"], "address": v["address"], "similarity": round(sim, 3), "matched": bool(_is_match(r)),
            "numbers_match": bool(digits), "sim_company": round(sc, 3), "sim_address": round(sa, 3),
            "n_train_receipts": v["n_train_receipts"]}

def vendor_lookup(company, address, tau=None):
    """For the app: the TRAIN vendor matching a predicted (company, address) at similarity >= tau (default RAG_TAU)
    on both fields with the same numbers, else None. Keys: company, address, similarity, matched (True),
    numbers_match, sim_company, sim_address, n_train_receipts. JSON-serialisable."""
    if not _is_match(_nearest(company, address), tau):
        return None
    return {**vendor_nearest(company, address), "matched": True}

def rag_correct(pred, tau=None):
    """Replace company/address with the matched TRAIN vendor's spelling; a field that already equals one of that
    vendor's train spellings is kept. Returns (corrected dict, changed fields, nearest tuple or None)."""
    r = _nearest(pred.get("company"), pred.get("address"))
    out, changed = dict(pred), []
    if _is_match(r, tau):
        v = r[3]
        for f in ("company", "address"):
            if str(pred.get(f) or "").strip() not in v[f + "_variants"]:
                out[f] = v[f]; changed.append(f)
    return out, changed, r

def rag_rescore(df, tau=None):
    rows = []
    for _, r in df.iterrows():
        gold = json.loads(r["expected"])
        pred = parse_json(r["prediction"])
        if pred is None:
            text, changed, near = r["prediction"], [], None
        else:
            new, changed, near = rag_correct(pred, tau)
            text = json.dumps(new, ensure_ascii=False)
        rows.append({"key": r["key"], "prediction": r["prediction"], "prediction_rag": text, "expected": r["expected"],
                     "rag_changed": ",".join(changed), "rag_similarity": np.nan if near is None else round(near[0], 3),
                     "rag_numbers_match": None if near is None else bool(near[4]),
                     "rag_vendor_company": "" if near is None else near[3]["company"],
                     "rag_vendor_address": "" if near is None else near[3]["address"], **score(text, gold)})
    return pd.DataFrame(rows)

# 2. Re-score the stored predictions of every configuration, with and without the vendor memory
RAG_BASES = {name: globals()[var] for name, var in (("plain", "plain_df"), ("prompted", "prompted_df"), ("finetuned", "ft_df"))
             if var in globals()}
rag_dfs = {}
for name, base in RAG_BASES.items():
    out = rag_rescore(base)
    same = (out["rag_changed"] == "").values
    ok = np.array_equal(out[METRICS].values[same], base[METRICS].values[same])
    out.to_csv(f"{RAG_DIR}/eval_{name}_rag.csv", index=False)
    rag_dfs[name] = out
    print(f"{name}: RAG changed company on {out['rag_changed'].str.contains('company').sum()} and address on "
          f"{out['rag_changed'].str.contains('address').sum()} of {len(out)} receipts | untouched rows re-score identically: {ok}")

_show = METRICS + ["company_sim", "address_sim"]
rag_comparison = pd.DataFrame({col: d[[m for m in _show if m in d]].mean() * 100
                               for name in RAG_BASES
                               for col, d in ((f"{name} (%)", RAG_BASES[name]), (f"{name}+RAG (%)", rag_dfs[name]))}).round(1)
rag_comparison.to_csv(f"{RAG_DIR}/rag_comparison.csv")
print("\n" + rag_comparison.to_string() + "\n(company_sim / address_sim = mean character similarity x 100)")

rag_tests = []
for name in sorted(rag_dfs, key=lambda n: n != "finetuned"):          # fine-tuned first
    for m in ("all4_norm", "all4_strict", "company_strict", "address_strict", "company_norm", "address_norm"):
        b, w, p = _mc(RAG_BASES[name][m].values, rag_dfs[name][m].values)
        rag_tests.append({"comparison": f"{name}+RAG vs {name}", "metric": m, "improved": b, "worse": w, "p": p})
        if name == "finetuned" or m.startswith("all4"):
            print(f"{name}+RAG vs {name} on {m}: {b} receipts improved, {w} got worse, exact McNemar p = {p:.2g}")
pd.DataFrame(rag_tests).to_csv(f"{RAG_DIR}/rag_mcnemar.csv", index=False)

# 3. POST-HOC sensitivity to TAU (TAU = 0.85 was fixed beforehand: report that row, not the best one)
_sweep = []
for name, base in RAG_BASES.items():
    _sweep.append({"tau": "off", "config": name, "receipts_changed": 0,
                   **{m: round(base[m].mean() * 100, 1) for m in ("company_strict", "address_strict", "all4_strict", "all4_norm")}})
    for tau in (0.70, 0.80, 0.85, 0.90, 0.95, 1.00):
        d = rag_rescore(base, tau)
        _sweep.append({"tau": tau, "config": name, "receipts_changed": int((d["rag_changed"] != "").sum()),
                       **{m: round(d[m].mean() * 100, 1) for m in ("company_strict", "address_strict", "all4_strict", "all4_norm")}})
rag_sweep = pd.DataFrame(_sweep)
rag_sweep.to_csv(f"{RAG_DIR}/rag_tau_sensitivity_POSTHOC.csv", index=False)
print("\nPOST-HOC sensitivity to TAU (exploratory only; the reported result uses TAU = %.2f, fixed a priori):" % RAG_TAU)
print(rag_sweep.to_string(index=False))

# 4. Diagnostic: how many test vendors already occur in the TRAIN labels? (test labels are only READ here, to describe
#    the test set; nothing is added to the memory and TAU was fixed before this)
_pool_c = [(k, g, _gnorm(g)) for k, g in ((k, _grams(k)) for k in sorted({v["mc"] for v in VENDOR_MEMORY}))]
_pool_a = [(k, g, _gnorm(g)) for k, g in ((k, _grams(k)) for k in sorted({v["ma"] for v in VENDOR_MEMORY}))]
def _best_field_ratio(q, pool):
    """Best difflib ratio against train strings with the same numbers (same rule as the vendor match)."""
    if not q:
        return 0.0
    gq, dq = _grams(q), re.findall(r"\d+", q); nq = _gnorm(gq)
    pool = [t for t in pool if not dq or re.findall(r"\d+", t[0]) == dq]
    return max((_ratio(q, t[0]) for t in sorted(pool, key=lambda t: -_cos(gq, nq, t[1], t[2]))[:RAG_TOPK]), default=0.0)
_raw_c = {c for v in VENDOR_MEMORY for c in v["company_variants"]}; _raw_a = {a for v in VENDOR_MEMORY for a in v["address_variants"]}
_mk_c = {v["mc"] for v in VENDOR_MEMORY}; _mk_a = {v["ma"] for v in VENDOR_MEMORY}
_ov = []
for s in test_subset:
    c, a = str(s["entities"]["company"] or "").strip(), str(s["entities"]["address"] or "").strip()
    near = _nearest(c, a)
    _ov.append({"key": s["key"], "company": c, "address": a,
                "company_verbatim": c in _raw_c, "company_same_ignoring_punct": _mkey(c) in _mk_c,
                "company_near_0.9": _best_field_ratio(_mkey(c), _pool_c) >= 0.9,
                "address_verbatim": a in _raw_a, "address_same_ignoring_punct": _mkey(a) in _mk_a,
                "address_near_0.9": _best_field_ratio(_mkey(a), _pool_a) >= 0.9,
                "vendor_similarity": np.nan if near is None else round(near[0], 3),
                "seen_vendor": bool(_is_match(near, 0.9))})
rag_overlap = pd.DataFrame(_ov)
rag_overlap.to_csv(f"{RAG_DIR}/test_vendor_overlap.csv", index=False)
_ov_cols = ["company_verbatim", "company_same_ignoring_punct", "company_near_0.9",
            "address_verbatim", "address_same_ignoring_punct", "address_near_0.9", "seen_vendor"]
print(f"\nTest-vendor overlap with the TRAIN labels ({len(rag_overlap)} test receipts, % of receipts):")
print(rag_overlap[_ov_cols].mean().mul(100).round(1).to_string())

# Where does each configuration score: vendors seen in training vs unseen vendors?
_seen = rag_overlap.set_index("key")["seen_vendor"].astype(bool)
_split = []
for name, d in list(RAG_BASES.items()) + [(f"{n}+RAG", rag_dfs[n]) for n in rag_dfs]:
    dd = d.set_index("key").loc[_seen.index]
    for grp, mask in (("seen vendor", _seen.values), ("unseen vendor", ~_seen.values)):
        if mask.sum():
            _split.append({"config": name, "test vendors": grp, "n": int(mask.sum()),
                           **{m: round(dd[m].values[mask].mean() * 100, 1) for m in ("company_strict", "address_strict", "all4_norm", "all4_strict")}})
rag_seen_split = pd.DataFrame(_split)
rag_seen_split.to_csv(f"{RAG_DIR}/seen_vs_unseen_vendor.csv", index=False)
print("\nAccuracy (%) on test receipts whose vendor IS / IS NOT in the train labels (similarity >= 0.9 on both fields, same numbers):")
print(rag_seen_split.to_string(index=False))

RAG_SUMMARY = {"method": "retrieval-augmented post-correction: vendor memory from TRAIN labels, char-trigram prefilter + difflib",
               "tau_a_priori": RAG_TAU, "train_vendors": len(VENDOR_MEMORY), "train_receipts_indexed": len(_index_keys),
               "test_receipts_with_seen_vendor_pct": round(float(rag_overlap["seen_vendor"].mean() * 100), 1),
               **{f"{n}_{m}": round(float(d[m].mean() * 100), 1) for n, d in list(RAG_BASES.items()) + [(f"{k}+RAG", v) for k, v in rag_dfs.items()]
                  for m in ("all4_norm", "all4_strict")}}
json.dump(RAG_SUMMARY, open(f"{RAG_DIR}/rag_summary.json", "w"), indent=1)
print("\n" + json.dumps(RAG_SUMMARY, indent=1))

# %%
# CELL: Training-steps experiment - checkpoint learning curve on the first 50 test receipts (about 2 min per checkpoint, GPU)
# Run BEFORE the Gradio app is live: while this cell runs, the "default" adapter temporarily holds checkpoint weights.
# Checkpoints are loaded into the live PEFT model (peft.set_peft_model_state_dict re-inserts the adapter name
# "default" into the saved keys); the final adapter is snapshotted first and restored bit for bit at the end.
import glob, os, re, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file as safe_load_file
from peft import set_peft_model_state_dict, load_peft_weights

LC_N = 50
LC_ADAPTER = "default"
LC_FINAL_STEP = TRAIN_STEPS
LC_RECEIPTS_PER_STEP = 4            # make_trainer: batch 1 x grad-accum 4
lc_subset = test_subset[:LC_N]
_LORA_RE = re.compile(rf"\.(lora_[A-Za-z_]+)\.{LC_ADAPTER}(\.|$)")
if getattr(globals().get("demo"), "is_running", False):
    print("WARNING: the Gradio app is live; it will serve CHECKPOINT weights until this cell finishes.")

def _mcnemar_exact(base, new):
    base, new = np.asarray(base), np.asarray(new)
    b, w = int(((base == 0) & (new == 1)).sum()), int(((base == 1) & (new == 0)).sum())
    n, k = b + w, min(b, w)
    return b, w, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))
_mc = globals().get("mcnemar", _mcnemar_exact)

# 1. Snapshot the final adapter and predict the checkpoint key names (peft saves keys with the adapter name removed)
_lora_params = {n: p for n, p in model.named_parameters() if _LORA_RE.search(n)}
assert _lora_params, f"no LoRA parameters for adapter {LC_ADAPTER!r}"
_final_snapshot = {n: p.detach().clone() for n, p in _lora_params.items()}
_expected_ckpt_keys = {_LORA_RE.sub(r".\1\2", n): n for n in _lora_params}
print(f"snapshot of the final adapter: {len(_final_snapshot)} LoRA tensors, "
      f"{sum(t.numel() for t in _final_snapshot.values()) / 1e6:.1f} M parameters | e.g. {next(iter(_expected_ckpt_keys))}")

_ckpts = []
for _p in glob.glob(f"{OUT}/outputs/checkpoint-*"):
    _m = re.search(r"checkpoint-(\d+)$", _p.rstrip("/"))
    if _m and os.path.isdir(_p):
        _ckpts.append((int(_m.group(1)), _p))
_ckpts.sort()
print("checkpoints found:", [s for s, _ in _ckpts], "| final step:", LC_FINAL_STEP)

def _load_ckpt(path):
    f = os.path.join(path, "adapter_model.safetensors")
    sd = safe_load_file(f, device="cpu") if os.path.exists(f) else load_peft_weights(path, device="cpu")
    if any("._orig_mod." in k for k in sd) and not any("._orig_mod." in k for k in _expected_ckpt_keys):
        sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}      # torch.compile wrapper names, if any
    return sd

def _apply(sd, tag):
    """Load a checkpoint state dict into the live 'default' adapter. False (and nothing loaded) if keys do not match."""
    missing = set(_expected_ckpt_keys) - set(sd)
    if missing:
        print(f"{tag}: SKIPPED, {len(missing)} LoRA tensors missing from the checkpoint (e.g. {sorted(missing)[:2]}); "
              f"checkpoint keys look like {sorted(sd)[:2]}")
        return False
    extra = [k for k in sd if k not in _expected_ckpt_keys]
    if extra:
        print(f"{tag}: ignoring {len(extra)} non-LoRA tensors in the checkpoint (e.g. {extra[:2]})")
    res = set_peft_model_state_dict(model, {k: sd[k] for k in _expected_ckpt_keys}, adapter_name=LC_ADAPTER)
    bad_missing = [k for k in res.missing_keys if k in _lora_params]
    if res.unexpected_keys or bad_missing:
        print(f"{tag}: load reported {len(res.unexpected_keys)} unexpected and {len(bad_missing)} missing LoRA keys")
        return False
    return True

lc_points, LC_KEYS_VERIFIED = [], None
try:
    # 2. Key-format check: the final checkpoint loaded through the same path must reproduce the in-memory adapter
    for _s, _p in _ckpts:
        if _s == LC_FINAL_STEP and _apply(_load_ckpt(_p), f"checkpoint-{_s}"):
            _d = max((p.detach().float() - _final_snapshot[n].float()).abs().max().item() for n, p in _lora_params.items())
            LC_KEYS_VERIFIED = _d == 0.0
            print(f"key-format check: checkpoint-{_s} reproduces the in-memory final adapter: {LC_KEYS_VERIFIED} (max |diff| {_d:.3g})")
    # 3. Evaluate each intermediate checkpoint on the same first LC_N test receipts
    for _s, _p in _ckpts:
        if _s >= LC_FINAL_STEP or not _apply(_load_ckpt(_p), f"checkpoint-{_s}"):
            continue
        _n_diff = sum(not torch.equal(p.detach(), _final_snapshot[n]) for n, p in _lora_params.items())
        print(f"checkpoint-{_s}: loaded, {_n_diff}/{len(_lora_params)} LoRA tensors differ from the final adapter")
        _df = evaluate(lc_subset, f"ckpt{_s}")
        if (_df["prediction"].values == ft_df["prediction"].values[:LC_N]).all():
            print(f"WARNING: checkpoint-{_s} predictions are identical to the final model's: check that its weights were used")
        lc_points.append((_s, _df))
finally:
    with torch.no_grad():
        for n, p in _lora_params.items():
            p.copy_(_final_snapshot[n])
    LC_RESTORE_OK = all(torch.equal(p.detach(), _final_snapshot[n]) for n, p in _lora_params.items())
    print("final adapter restored (weights bit-identical):", LC_RESTORE_OK)

# 4. Verify the restore end to end: regenerate 2 receipts and compare with the stored fine-tuned predictions
FastVisionModel.for_inference(model)
_regen_ok = []
for i in range(2):
    _now = generate_from_image(test_subset[i]["image"])
    _regen_ok.append(_now == ft_df["prediction"].iloc[i])
    if not _regen_ok[-1]:
        print(f"receipt {i}: regenerated output differs from ft_df\n  now: {_now!r}\n  was: {ft_df['prediction'].iloc[i]!r}")
LC_RESTORE_OK = bool(LC_RESTORE_OK and all(_regen_ok))
print("restore verified by regenerating 2 receipts:", all(_regen_ok))
if LC_KEYS_VERIFIED is False:
    print("WARNING: the final checkpoint differs from the in-memory adapter (another training run?); treat the curve with care.")
if not lc_points:
    print(f"No intermediate checkpoint was evaluated (none under {OUT}/outputs, or all skipped): the curve has only step 0 and the final step.")
del _final_snapshot; torch.cuda.empty_cache()

# 5. Learning curve: step 0 = plain model, intermediate checkpoints, final adapter (same LC_N receipts, same scorer)
_final = ft_df.iloc[:LC_N]
_pts = sorted([(0, plain_df.iloc[:LC_N])] + lc_points + [(LC_FINAL_STEP, _final)], key=lambda t: t[0])
learning_curve = pd.DataFrame([{"step": s, "receipts_seen": s * LC_RECEIPTS_PER_STEP, "n_test": len(d),
                                **(d[METRICS].mean() * 100).round(1).to_dict(),
                                "p_vs_final_all4_norm": np.nan if s == LC_FINAL_STEP else _mc(d["all4_norm"].values, _final["all4_norm"].values)[2]}
                               for s, d in _pts])
learning_curve.to_csv(f"{OUT}/learning_curve.csv", index=False)
print(learning_curve[["step", "receipts_seen", "valid_json", "all4_norm", "all4_strict"]
                     + [f"{f}_norm" for f in FIELDS] + ["p_vs_final_all4_norm"]].to_string(index=False))
print("Caveat: the points are snapshots of ONE run with a cosine LR schedule, not separately trained shorter runs; "
      f"n = {LC_N}, so a few points of difference is noise (see p_vs_final, exact McNemar).")

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(learning_curve["step"], learning_curve["all4_norm"], "o-", lw=2, label="all four fields, normalized")
ax.plot(learning_curve["step"], learning_curve["all4_strict"], "s--", lw=2, label="all four fields, strict")
for f in FIELDS:
    ax.plot(learning_curve["step"], learning_curve[f"{f}_norm"], lw=0.9, alpha=0.6, label=f"{f}, normalized")
ax.set_xticks(learning_curve["step"])
ax.set_xlabel(f"optimizer steps (each = {LC_RECEIPTS_PER_STEP} training receipts; step 0 = plain model)")
ax.set_ylabel(f"% of the first {LC_N} test receipts"); ax.set_ylim(0, 100)
ax.set_title("Learning curve: Gemma 4 E2B LoRA on SROIE"); ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/learning_curve.png", dpi=150); plt.show()

# %%
# CELL: Aggregate background experiments (exp_*) with the main run - one table + one chart (seconds; about 40 s more if it times the main run)
import glob, os, re, json, time, math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

AGG_TIME_MAIN = True     # time AGG_TIME_N receipts per main configuration for s/receipt (skipped if the adapter restore failed)
AGG_TIME_N = 5
_NAMES = {
    "tag": ("tag", "name", "exp", "experiment", "exp_name", "run"),
    "model": ("model", "model_name", "MODEL_NAME", "base_model"),
    "rank": ("lora_r", "rank", "r", "LORA_R"),
    "steps": ("steps", "train_steps", "max_steps", "TRAIN_STEPS"),
    "train_fraction": ("train_frac", "train_fraction", "data_fraction", "frac", "fraction", "TRAIN_FRAC"),
    "train_receipts": ("train_receipts", "n_train", "num_train", "train_size", "n_train_receipts"),
    "lr": ("lr", "learning_rate", "LEARNING_RATE"),
    "s_per_step": ("s_per_step", "sec_per_step", "s_step", "seconds_per_step"),
    "s_per_receipt": ("s_per_receipt", "sec_per_receipt", "s_receipt", "seconds_per_receipt"),
    "peak_gb": ("peak_gb", "peak_gpu_gb", "peak_memory_gb", "peak_mem_gb", "peak_gpu_mem_gb", "peak_gpu_memory_gb", "max_memory_gb"),
    "train_runtime": ("train_runtime", "train_runtime_s", "train_seconds"),
    "visual_budget": ("visual_budget", "max_soft_tokens", "MAX_SOFT_TOKENS"),
    "load_in_4bit": ("load_in_4bit", "LOAD_IN_4BIT"),
}
_MCOLS = ["all4_norm", "all4_strict"] + [f"{f}_norm" for f in FIELDS] + ["valid_json"]

def _walk(d, path=()):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from _walk(v, path + (k,))
        else:
            yield path + (k,), v

def _lookup(summary, field, config=None, other_configs=()):
    """First value for any alias of `field` anywhere in summary.json, preferring entries under this config.
    Handles {"s_per_receipt": 2.1}, {"finetuned": {"s_per_receipt": 2.1}} and {"s_per_receipt": {"finetuned": 2.1}}."""
    names, best = _NAMES[field], None
    for path, v in _walk(summary):
        if v is None or v == "":
            continue
        if path[-1] in names:
            name, ctx = path[-1], path[:-1]
        elif len(path) >= 2 and path[-2] in names and path[-1] == config:
            name, ctx = path[-2], path
        else:
            continue
        if any(p in other_configs for p in ctx):
            continue
        rank = (0 if config in ctx else 1, len(path), names.index(name))
        if best is None or rank < best[0]:
            best = (rank, v)
    return np.nan if best is None else best[1]

def _num(v):
    if isinstance(v, (bool, int, float, np.integer, np.floating)):
        return float(v)
    m = re.findall(r"-?\d+(?:\.\d+)?", str(v))
    return float(m[-1]) if m else np.nan          # e.g. "5 smoke + 157" -> 157

def _bool(v):
    return np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip().lower() in ("true", "1", "1.0", "yes")

def _metrics(df):
    return {m: round(float(df[m].astype(float).mean()) * 100, 1) if m in df else np.nan for m in _MCOLS}

def _mcnemar_exact(base, new):
    base, new = np.asarray(base), np.asarray(new)
    b, w = int(((base == 0) & (new == 1)).sum()), int(((base == 1) & (new == 0)).sum())
    n, k = b + w, min(b, w)
    return b, w, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))
_mc = globals().get("mcnemar", _mcnemar_exact)

def _time_main(plain=False, instruction=None, n=AGG_TIME_N):
    FastVisionModel.for_inference(model); ts = []
    for s in test_subset[:n]:
        t0 = time.time()
        if plain:
            with model.disable_adapter():
                generate_from_image(s["image"], instruction or INSTRUCTION)
        else:
            generate_from_image(s["image"], instruction or INSTRUCTION)
        ts.append(time.time() - t0)
    return round(float(np.median(ts)), 2)

# 1. The main run from this notebook's dataframes
_rows = []
_peak = round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else np.nan
_s_step = (stats.metrics.get("train_runtime", np.nan) / TRAIN_STEPS) if "stats" in globals() and getattr(stats, "metrics", None) else np.nan
_can_time = AGG_TIME_MAIN and globals().get("LC_RESTORE_OK", True)
for cfg, var, kw in (("plain", "plain_df", {"plain": True}), ("prompted", "prompted_df", {"plain": True, "instruction": PROMPT_DETAILED}),
                     ("finetuned", "ft_df", {})):
    if var not in globals():
        continue
    _s_rec = np.nan
    if _can_time:
        try:
            _s_rec = _time_main(**kw)
        except Exception as e:
            print(f"timing of main/{cfg} skipped: {e!r}"[:200])
    _ft = cfg == "finetuned"
    _rows.append({"tag": "main", "model": MODEL_NAME, "rank": LORA_R if _ft else np.nan, "steps": TRAIN_STEPS if _ft else 0,
                  "train_fraction": 1.0 if _ft else np.nan, "lr": LEARNING_RATE if _ft else np.nan, "config": cfg,
                  "n_test": len(globals()[var]), **_metrics(globals()[var]), "s_per_receipt": _s_rec,
                  "s_per_step": round(_s_step, 2) if _ft else np.nan, "peak_gb": _peak,
                  "visual_budget": MAX_SOFT_TOKENS, "load_in_4bit": LOAD_IN_4BIT, "comparable": True, "source": "this notebook"})

# 2. Background experiments: {OUT}/exp_*/summary.json + {OUT}/exp_*/eval_<config>.csv
_ft_by_key = ft_df.assign(key=ft_df["key"].astype(str)).set_index("key") if "ft_df" in globals() else None
_exp_dirs = sorted(d for d in glob.glob(f"{OUT}/exp_*") if os.path.isdir(d))
if not _exp_dirs:
    print(f"No background experiments under {OUT}/exp_*/ yet: showing the main run only.")
for d in _exp_dirs:
    try:
        summ, sp = {}, os.path.join(d, "summary.json")
        if os.path.exists(sp):
            try:
                summ = json.load(open(sp))
            except Exception as e:
                print(f"{d}: unreadable summary.json ({e!r})")
        csvs = sorted(glob.glob(os.path.join(d, "eval_*.csv")))
        configs = [os.path.basename(c)[len("eval_"):-len(".csv")] for c in csvs]
        for c, cfg in (zip(csvs, configs) if csvs else [(None, "(no eval csv yet)")]):
            others = set(configs) - {cfg}
            get = lambda field: _lookup(summ, field, cfg, others)
            e = pd.read_csv(c) if c else pd.DataFrame()
            tag = get("tag"); tag = os.path.basename(d).removeprefix("exp_") if isinstance(tag, float) and math.isnan(tag) else str(tag)
            steps, frac, n_train = _num(get("steps")), _num(get("train_fraction")), _num(get("train_receipts"))
            if np.isnan(frac) and not np.isnan(n_train):
                frac = n_train / len(train_data)
            s_step = _num(get("s_per_step"))
            if np.isnan(s_step) and not np.isnan(_num(get("train_runtime"))) and steps:
                s_step = _num(get("train_runtime")) / steps
            budget, load4 = _num(get("visual_budget")), _bool(get("load_in_4bit"))
            row = {"tag": tag, "model": get("model"), "rank": _num(get("rank")), "steps": steps, "train_fraction": frac,
                   "lr": _num(get("lr")), "config": cfg, "n_test": len(e), **_metrics(e),
                   "s_per_receipt": _num(get("s_per_receipt")), "s_per_step": s_step, "peak_gb": _num(get("peak_gb")),
                   "visual_budget": budget, "load_in_4bit": load4,
                   "comparable": (bool((np.isnan(budget) or budget == MAX_SOFT_TOKENS) and (pd.isna(load4) or load4 == LOAD_IN_4BIT))
                                  if c else np.nan),
                   "source": os.path.relpath(c or d, OUT)}
            if len(e) and _ft_by_key is not None and "all4_norm" in e:
                eb = e.assign(key=e["key"].astype(str)).drop_duplicates("key").set_index("key")
                common = [k for k in eb.index if k in _ft_by_key.index]
                if common:
                    row["main_ft_all4_norm_same_keys"] = round(float(_ft_by_key.loc[common, "all4_norm"].mean()) * 100, 1)
                    b, w, p = _mc(_ft_by_key.loc[common, "all4_norm"].values, eb.loc[common, "all4_norm"].values)
                    row["vs_main_ft_improved/worse"] = f"{b}/{w}"; row["p_vs_main_ft_all4_norm"] = p
            _rows.append(row)
    except Exception as e:
        print(f"{d}: skipped ({e!r})"[:300])

experiments = pd.DataFrame(_rows)
experiments.to_csv(f"{OUT}/experiments_table.csv", index=False)
_cols = ["tag", "model", "rank", "steps", "train_fraction", "lr", "config", "n_test", "all4_norm", "all4_strict"] \
        + [f"{f}_norm" for f in FIELDS] + ["s_per_receipt", "s_per_step", "peak_gb", "comparable",
                                          "main_ft_all4_norm_same_keys", "vs_main_ft_improved/worse", "p_vs_main_ft_all4_norm"]
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(experiments[[c for c in _cols if c in experiments]].to_string(index=False))
print("peak_gb for the main run is this runtime's maximum so far (training + evaluation). 'comparable' = same visual "
      "budget and load setting as the main run; main_ft_all4_norm_same_keys = the main fine-tuned model on the same receipts.")

_plot = experiments.dropna(subset=["all4_norm"]) if "all4_norm" in experiments else experiments.iloc[0:0]
if len(_plot):
    _x, _w = np.arange(len(_plot)), 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(_plot) + 2), 4.2))
    ax.bar(_x - _w / 2, _plot["all4_norm"], _w, label="all four fields, normalized")
    ax.bar(_x + _w / 2, _plot["all4_strict"], _w, label="all four fields, strict")
    ax.set_xticks(_x)
    ax.set_xticklabels([f"{r.tag}\n{r.config}\n(n={int(r.n_test)})" for r in _plot.itertuples()], fontsize=8)
    ax.set_ylabel("% of test receipts"); ax.set_ylim(0, 100); ax.legend(fontsize=8)
    ax.set_title("Main run and background experiments (Gemma 4 on SROIE)")
    plt.tight_layout(); plt.savefig(f"{OUT}/experiments.png", dpi=150); plt.show()
