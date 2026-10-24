# === PASTE THIS WHOLE CELL INTO THE ORIGINAL COLAB NOTEBOOK (new cell at the bottom) AND RUN IT (about 1 min) ===
# Runs the polished "Receipt to Expense" UI inside Colab (no laptop hop), with the best model (budget 1120),
# a fast fine-tuned-only scan, split the bill, expenses, and the new logo. Prints a NEW public link (PRO_URL).
# The old public link keeps working (its "Fill items" / "By items" are fixed too).

# --- PART 1: restore the old app's _rows() helper (a later cell reused the name, which broke "Fill items")
def _rows(df):
    if df is None:
        return []
    if isinstance(df, dict):
        df = pd.DataFrame(df.get("data") or [], columns=(df.get("headers") or None))
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return [[_s(x) for x in r] for r in df.itertuples(index=False)]

# --- PART 2: best version = the budget-1120 adapter (92% of fields, 76% of receipts all-correct), swapped in place
import re as _re2
from safetensors.torch import load_file as _load_sd
from peft import set_peft_model_state_dict as _set_sd
_LRE = _re2.compile(r"\.(lora_[A-Za-z_]+)\.default(\.|$)")
_exp = {_LRE.sub(r".\1\2", n): n for n, p in model.named_parameters() if _LRE.search(n)}
_sd = _load_sd(f"{OUT}/exp_e2b_r32_s157_b1120/lora_adapter/adapter_model.safetensors", device="cpu")
assert _exp and not (set(_exp) - set(_sd)), "LoRA keys differ: nothing was changed"
with GEN_LOCK:
    _res = _set_sd(model, {k: _sd[k] for k in _exp}, adapter_name="default")
    MAX_SOFT_TOKENS = 1120
    processor.image_processor.max_soft_tokens = 1120
    FastVisionModel.for_inference(model)
print("best adapter loaded:", not _res.unexpected_keys, "| visual budget", MAX_SOFT_TOKENS)

# --- PART 3: the polished app (everything scoped inside _build_pro, so no global name is overwritten)
def _build_pro():
    import html as _html, json as _json, os as _os, re as _re, threading as _th, time as _time
    import gradio as _gr
    import pandas as _pd
    esc = lambda v: _html.escape(str(v))
    LOGO = '''<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 64 64" role="img" aria-label="Receipt to Expense"><defs><linearGradient id="r2eLogoG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#14c48b"/><stop offset="1" stop-color="#047857"/></linearGradient></defs><rect x="0" y="0" width="64" height="64" rx="16" ry="16" fill="url(#r2eLogoG)"/><path d="M16 11h24v32l-4-3-4 3-4-3-4 3-4-3-4 3z" fill="#ffffff"/><rect x="20" y="16.5" width="16" height="3" rx="1.5" fill="#0f9f6e" opacity=".35"/><rect x="20" y="22.5" width="10" height="3" rx="1.5" fill="#0f9f6e" opacity=".35"/><rect x="20" y="29" width="16" height="3.4" rx="1.7" fill="#0f172a"/><circle cx="42" cy="42" r="10.5" fill="#047857"/><circle cx="42" cy="42" r="8" fill="#ffffff"/><path d="M38.2 42.3l2.7 2.7 5.2-5.6" fill="none" stroke="#0f9f6e" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'''
    fav = f"{OUT}/r2e_logo.svg"
    open(fav, "w").write(LOGO.replace('width="44" height="44"', 'width="64" height="64"'))
    HF = "https://huggingface.co/owaisnoe/gemma4-e2b-sroie-lora"
    FIELDS4 = ["company", "date", "address", "total"]
    expenses_p, group_p = [], {"balances": {}, "history": []}

    def s_(v):
        return "" if v is None else (v if isinstance(v, str) else _json.dumps(v, ensure_ascii=False)).strip()

    def tot_(x):
        nums = _re.findall(r"\d+(?:\.\d+)?", str(x or "").replace(",", ""))
        return f"{float(nums[-1]):.2f}" if nums else ""

    def money(v):
        return f"{v:,.2f}"

    def same(k, a, b):
        return NORMALIZERS[k](a) != "" and NORMALIZERS[k](a) == NORMALIZERS[k](b)

    EMPTY = ('<div class="pc-card pc-empty"><div class="pc-empty-logo">' + LOGO + '</div><div class="pc-empty-t">No receipt yet</div>'
             '<div class="pc-muted">Snap or upload a receipt, then press <b>Scan receipt</b>. The fine-tuned model reads the '
             'merchant, date, address and total.</div></div>')

    def card(company, date, address, total, secs):
        flags = [n for n, ok in (("total", bool(tot_(total))), ("date", bool(_re.search(r"\d", date or ""))),
                                 ("merchant", bool(company)), ("address", bool(address))) if not ok]
        chips = [f'<span class="pc-chip pc-warn">&#9888; Check {esc(", ".join(flags))}</span>' if flags
                 else '<span class="pc-chip pc-ok">&#10003; All 4 fields read</span>',
                 '<span class="pc-chip">Fine-tuned model v2</span>', f'<span class="pc-chip">{secs:.1f} s</span>']
        return (f'<div class="pc-card pc-summary"><div class="pc-row"><div class="pc-avatar">{esc((company[:1] or "?").upper())}</div>'
                f'<div class="pc-grow"><div class="pc-merchant">{esc(company) or "Unknown merchant"}</div>'
                f'<div class="pc-muted">{esc(date) or "no date"}</div></div>'
                f'<div class="pc-amount"><span class="pc-cur">Total</span>{esc(tot_(total) or total or "-")}</div></div>'
                f'<div class="pc-address">{esc(address)}</div><div class="pc-chips">{"".join(chips)}</div></div>')

    def scan(image):
        if image is None:
            raise _gr.Error("Add a receipt photo first (upload, camera or a sample).")
        t = _time.time()
        try:
            raw, _ = _generate(image, adapter=True)            # fine-tuned only; GEN_LOCK inside
        except Exception as e:
            raise _gr.Error(f"The model failed on this photo ({type(e).__name__}). Try another one.")
        f = parse_json(raw) or {}
        c, d, a, tt = (s_(f.get(k)) for k in FIELDS4)
        if not f:
            _gr.Warning("The model's answer was not valid JSON; type the fields by hand.")
        return card(c, d, a, tt, _time.time() - t), c, d, tt, a, raw, ""

    def vendor(company, address):
        if not company:
            return ""
        try:
            v = vendor_lookup(company, address)
        except Exception:
            return ""
        return ('<div class="pc-vendor pc-ok">&#10003; Known vendor: seen in the training receipts</div>' if v else
                '<div class="pc-vendor">New vendor: not in the training receipts</div>')

    def compare(image, company, date, total, address):
        if image is None:
            raise _gr.Error("Scan a receipt first.")
        praw, _ = _generate(image, adapter=False)
        p = parse_json(praw) or {}
        mine = {"company": company, "date": date, "address": address, "total": total}
        rows, agree = [], 0
        for k in FIELDS4:
            ok = same(k, s_(mine[k]), s_(p.get(k)))
            agree += ok
            rows.append(f'<tr><th>{k.title()}</th><td>{esc(mine[k])}</td><td>{esc(s_(p.get(k)))}</td>'
                        f'<td class="{"r2e-yes" if ok else "r2e-no"}">{"&#10003; match" if ok else "&#10007; differs"}</td></tr>')
        return (f'<div class="pc-banner">Fine-tuned and plain agree on <b>{agree}/4</b> fields. Where they agree, the field was right '
                f'83-100% of the time on our test receipts; where they differ, double-check.</div>'
                f'<div class="r2e-scroll"><table class="r2e-table"><tr><th>Field</th><th>Fine-tuned (ours)</th><th>Plain Gemma 4</th><th></th></tr>'
                f'{"".join(rows)}</table></div>'), praw

    def autofill(image):
        global MAX_SOFT_TOKENS
        if image is None:
            raise _gr.Error("Scan a receipt first.")
        with GEN_LOCK:                                         # line items: plain model at budget 560, as evaluated
            b, MAX_SOFT_TOKENS = MAX_SOFT_TOKENS, 560
            try:
                items, raw, secs = extract_line_items(image, use_adapter=False)
            finally:
                MAX_SOFT_TOKENS = b
        rows = [[s_(it.get("name")), s_(it.get("price")), "all"] for it in (items or []) if isinstance(it, dict)]
        if not rows:
            _gr.Warning("No line items found: add rows by hand, or split equally.")
        return _pd.DataFrame(rows, columns=["item", "price", "shared by"]), _gr.Radio(value="By items")

    def names_(x):
        out = []
        for n in _re.split(r"[,;\n]+", str(x or "")):
            n = n.strip()
            if n and n.lower() not in {y.lower() for y in out}:
                out.append(n)
        return out

    def amt(x):
        v = tot_(s_(x))
        return float(v) if v else None

    def rows_(df):
        if df is None:
            return []
        if isinstance(df, dict):
            df = _pd.DataFrame(df.get("data") or [], columns=(df.get("headers") or None))
        if not isinstance(df, _pd.DataFrame):
            df = _pd.DataFrame(df)
        return [[s_(x) for x in r] for r in df.itertuples(index=False)]

    def split(people, payer, mode, items_df, total, company):
        names = names_(people)
        if len(names) < 2:
            raise _gr.Error("Add at least two people, separated by commas.")
        payer_n = next((n for n in names if n.lower() == s_(payer).lower()), None)
        if payer_n is None:
            raise _gr.Error(f"'Paid by' must be one of: {', '.join(names)}.")
        tt = amt(total)
        if tt is None or tt <= 0:
            raise _gr.Error("Scan a receipt (or type the total) first.")
        shares, items_of, notes, others, items_sum = {n: 0.0 for n in names}, {n: [] for n in names}, [], 0.0, None
        if mode == "By items":
            priced = 0
            for r in rows_(items_df):
                r = r + ["", "", ""]
                item, price, who = r[0], amt(r[1]), r[2]
                if price is None:
                    if item:
                        notes.append(f"skipped '{item}' (no price)")
                    continue
                wanted = [w.strip().lower() for w in _re.split(r"[,;/&+]|\band\b", who) if w.strip()]
                ppl = names if not wanted or wanted[0] in ("all", "everyone", "everybody") else [n for n in names if n.lower() in wanted]
                if not ppl:
                    notes.append(f"'{item}': nobody called '{who}', so it is shared by everyone")
                    ppl = names
                for n in ppl:
                    shares[n] += price / len(ppl); items_of[n].append(item or "item")
                priced += 1
            if not priced:
                raise _gr.Error("No priced items: press 'Auto-fill items', type them, or split equally.")
            items_sum = sum(shares.values())
            others = round(tt - items_sum, 2)
            if abs(others) >= 0.01:
                for n in names:
                    shares[n] += others * (shares[n] / items_sum) if items_sum else others / len(names)
        else:
            for n in names:
                shares[n], items_of[n] = tt / len(names), ["equal share"]
        rounded = {n: round(v, 2) for n, v in shares.items()}
        rounded[payer_n] = round(rounded[payer_n] + (tt - sum(rounded.values())), 2)
        left = round(tt - sum(rounded.values()), 2)
        warn = ""
        if items_sum is not None and abs(others) >= 0.01:
            big = abs(others) > 0.25 * tt
            warn = (f'<div class="pc-banner {"pc-banner-warn" if big else ""}">Others (tax, service, rounding'
                    f'{", discount" if others < 0 else ""}): <b>{money(others)}</b>, shared in proportion.'
                    + (" Make sure this amount is correct." if big else "") + "</div>")
        people_html = "".join(
            f'<div class="pc-person"><div class="pc-avatar pc-sm">{esc(n[:1].upper())}</div><div class="pc-grow">'
            f'<div class="pc-pname">{esc(n)}</div><div class="pc-muted">{esc(", ".join(items_of[n][:4]))}</div></div>'
            f'<div class="pc-right"><div class="pc-pamt">{money(rounded[n])}</div>'
            f'<div class="{"pc-paid" if n == payer_n else "pc-owes"}">{"paid the bill" if n == payer_n else "owes " + esc(payer_n)}</div></div></div>'
            for n in names)
        tick = "&#10003;" if abs(left) < 0.005 else ""
        out = (f'<div class="pc-card"><div class="pc-row"><div class="pc-grow"><div class="pc-merchant">{esc(s_(company)) or "This receipt"}</div>'
               f'<div class="pc-muted">{len(names)} people &middot; {"by item" if mode == "By items" else "split equally"} &middot; paid by {esc(payer_n)}</div></div>'
               f'<div class="pc-amount"><span class="pc-cur">Total</span>{money(tt)}</div></div>{warn}{people_html}'
               f'<div class="pc-left">Left to split <b>{money(left)}</b> <span class="pc-tick">{tick}</span></div>'
               + (f'<div class="pc-muted">{esc("; ".join(notes))}</div>' if notes else "") + "</div>")
        return out, {"label": s_(company) or "receipt", "payer": payer_n, "total": tt, "shares": rounded}

    def settle(bal):
        cred = sorted([[n, v] for n, v in bal.items() if v > 0.005], key=lambda x: -x[1])
        debt = sorted([[n, -v] for n, v in bal.items() if v < -0.005], key=lambda x: -x[1])
        out, i, j = [], 0, 0
        while i < len(debt) and j < len(cred):
            a_ = round(min(debt[i][1], cred[j][1]), 2)
            if a_ >= 0.01:
                out.append((debt[i][0], cred[j][0], a_))
            debt[i][1] -= a_; cred[j][1] -= a_
            if debt[i][1] < 0.005: i += 1
            if cred[j][1] < 0.005: j += 1
        return out

    def group_html():
        bal, hist = group_p["balances"], group_p["history"]
        if not hist:
            return '<div class="pc-card pc-empty-sm">No shared bills yet. Calculate a split, then add it to the group.</div>'
        tiles = "".join(f'<div class="pc-kpi"><div class="pc-kl">{esc(n)}</div><div class="pc-kv {"pc-pos" if v > 0 else "pc-neg" if v < 0 else ""}">'
                        f'{"+" if v > 0 else ""}{money(v)}</div></div>' for n, v in sorted(bal.items(), key=lambda x: -x[1]))
        moves = "".join(f'<div class="pc-move"><b>{esc(a)}</b> pays <b>{esc(b)}</b><span>{money(x)}</span></div>'
                        for a, b, x in settle(bal)) or '<div class="pc-move">All settled up &#10003;</div>'
        return (f'<div class="pc-card"><div class="pc-h">Group balances <span class="pc-muted">({len(hist)} bill{"s" if len(hist) != 1 else ""})</span></div>'
                f'<div class="pc-kpis">{tiles}</div><div class="pc-h">Settle up with the fewest payments</div>{moves}</div>')

    def add_group(state):
        if not state:
            raise _gr.Error("Calculate a split first.")
        bal = group_p["balances"]
        for n, sh in state["shares"].items():
            bal[n] = round(bal.get(n, 0.0) - sh, 2)
        bal[state["payer"]] = round(bal.get(state["payer"], 0.0) + state["total"], 2)
        group_p["history"].append({"bill": state["label"], "paid_by": state["payer"], "total": state["total"], **state["shares"]})
        _gr.Info(f"Added {state['label']} to the group.")
        return group_html()

    def reset_g():
        group_p["balances"].clear(); group_p["history"].clear()
        return group_html()

    def exp_view():
        df = _pd.DataFrame(expenses_p, columns=["merchant", "date", "total", "address"])
        spend = sum(float(tot_(x) or 0) for x in df["total"])
        path = f"{OUT}/pro_expenses.csv"; df.to_csv(path, index=False)
        last = esc(df.iloc[-1]["merchant"] or "-") if len(df) else "-"
        kpis = (f'<div class="pc-kpis"><div class="pc-kpi"><div class="pc-kl">Total spend</div><div class="pc-kv">{money(spend)}</div></div>'
                f'<div class="pc-kpi"><div class="pc-kl">Receipts</div><div class="pc-kv">{len(df)}</div></div>'
                f'<div class="pc-kpi"><div class="pc-kl">Last added</div><div class="pc-kv pc-small">{last}</div></div></div>')
        return kpis, df, _gr.DownloadButton(value=path if len(df) else None, interactive=bool(len(df)))

    def add_exp(company, date, total, address):
        if not any(s_(x) for x in (company, date, total, address)):
            raise _gr.Error("Nothing to add yet: scan a receipt first.")
        expenses_p.append({"merchant": s_(company), "date": s_(date), "total": s_(total), "address": s_(address)})
        _gr.Info(f"Added {s_(company) or 'expense'} ({s_(total)}).")
        return exp_view()

    def undo_exp():
        if expenses_p:
            expenses_p.pop()
        return exp_view()

    theme = _gr.themes.Base(primary_hue="emerald", neutral_hue="slate", radius_size="lg",
                            font=[_gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
        body_background_fill="#f5f6f8", block_background_fill="#ffffff", block_border_color="#e6e8ec", block_border_width="1px",
        block_radius="16px", block_shadow="0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06)", block_label_text_weight="600",
        button_primary_background_fill="#0f9f6e", button_primary_background_fill_hover="#0b8a5f", button_primary_text_color="#ffffff",
        button_large_radius="12px", button_small_radius="10px", input_radius="10px", input_background_fill="#f8fafc")
    css = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important}
footer {display: none !important}
.pc-appbar {display:flex; align-items:center; gap:14px; padding:14px 6px 6px}
.pc-appbar svg {flex:none; filter:drop-shadow(0 6px 14px rgba(16,185,129,.30))}
.pc-title {font-size:1.45rem; font-weight:700; letter-spacing:-.01em; color:#0f172a}
.pc-sub {color:#64748b; font-size:.92rem}
.pc-pill {margin-left:auto; font-size:.78rem; font-weight:600; color:#047857; background:#ecfdf5; border:1px solid #a7f3d0; padding:.25rem .7rem; border-radius:999px}
.pc-card {background:#fff; border:1px solid #e6e8ec; border-radius:16px; padding:18px 18px 14px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06)}
.pc-empty {text-align:center; padding:30px 18px} .pc-empty-logo svg {width:56px; height:56px; opacity:.9}
.pc-empty-t {font-weight:700; font-size:1.05rem; margin:.4rem 0 .3rem} .pc-empty-sm {color:#64748b; text-align:center; padding:18px}
.pc-row {display:flex; align-items:center; gap:14px} .pc-grow {flex:1; min-width:0}
.pc-avatar {width:46px; height:46px; border-radius:14px; background:#ecfdf5; color:#047857; font-weight:700; font-size:1.2rem; display:flex; align-items:center; justify-content:center; flex:none}
.pc-avatar.pc-sm {width:36px; height:36px; border-radius:11px; font-size:1rem}
.pc-merchant {font-weight:700; font-size:1.08rem; color:#0f172a; overflow-wrap:anywhere} .pc-muted {color:#64748b; font-size:.88rem}
.pc-amount {font-size:1.9rem; font-weight:800; color:#0f172a; text-align:right; line-height:1.05; letter-spacing:-.02em}
.pc-cur {display:block; font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8}
.pc-address {color:#334155; font-size:.9rem; margin:.7rem 0 .2rem; padding:.6rem .75rem; background:#f8fafc; border-radius:10px}
.pc-chips {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.6rem}
.pc-chip {font-size:.78rem; font-weight:600; padding:.22rem .65rem; border-radius:999px; background:#f1f5f9; color:#334155; border:1px solid #e2e8f0}
.pc-chip.pc-ok {background:#ecfdf5; color:#047857; border-color:#a7f3d0} .pc-chip.pc-warn {background:#fffbeb; color:#b45309; border-color:#fde68a}
.pc-vendor {margin:.5rem 0 0; font-size:.85rem; font-weight:600; color:#334155; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:12px; padding:.5rem .8rem}
.pc-vendor.pc-ok {background:#ecfdf5; color:#047857; border-color:#a7f3d0}
.pc-banner {background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; border-radius:10px; padding:.55rem .75rem; margin:.7rem 0; font-size:.88rem}
.pc-banner-warn {background:#fffbeb; border-color:#fde68a; color:#92400e}
.pc-person {display:flex; align-items:center; gap:12px; padding:.6rem 0; border-top:1px solid #f1f5f9}
.pc-pname {font-weight:600; color:#0f172a} .pc-right {text-align:right} .pc-pamt {font-weight:700; font-size:1.05rem}
.pc-owes {color:#b45309; font-size:.8rem; font-weight:600} .pc-paid {color:#047857; font-size:.8rem; font-weight:600}
.pc-left {margin-top:.5rem; font-size:.88rem; color:#334155} .pc-tick {color:#10b981; font-weight:800}
.pc-h {font-weight:700; color:#0f172a; margin:.2rem 0 .6rem}
.pc-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem; margin-bottom:.8rem}
.pc-kpi {background:#f8fafc; border:1px solid #eef2f6; border-radius:12px; padding:.7rem .85rem}
.pc-kl {font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; color:#94a3b8; font-weight:600}
.pc-kv {font-size:1.35rem; font-weight:800; color:#0f172a; margin-top:.15rem} .pc-kv.pc-small {font-size:.95rem; overflow-wrap:anywhere}
.pc-pos {color:#047857} .pc-neg {color:#b45309}
.pc-move {display:flex; justify-content:space-between; padding:.5rem 0; border-top:1px solid #f1f5f9; color:#334155} .pc-move span {font-weight:700; color:#0f172a}
.pc-foot {text-align:center; color:#94a3b8; font-size:.8rem; padding:12px 0 4px} .pc-foot a {color:#047857}
.r2e-scroll {overflow-x:auto} .r2e-table {width:100%; border-collapse:collapse; font-size:.88rem}
.r2e-table th, .r2e-table td {padding:.45rem .5rem; border-bottom:1px solid #eef2f6; text-align:left; vertical-align:top; overflow-wrap:anywhere}
.r2e-yes {color:#10b981; font-weight:700} .r2e-no {color:#ef4444; font-weight:700}
@media (max-width: 640px) { .pc-amount {font-size:1.5rem} .pc-title {font-size:1.2rem} .pc-pill {display:none} }
"""
    with _gr.Blocks(title="Receipt to Expense") as app:
        _gr.HTML(f'<div class="pc-appbar">{LOGO}<div><div class="pc-title">Receipt to Expense</div>'
                 '<div class="pc-sub">Snap a receipt &middot; check it &middot; split it. Fine-tuned Gemma 4 E2B, open model.</div></div>'
                 '<div class="pc-pill">&#9679; Live</div></div>')
        split_state = _gr.State(None)
        with _gr.Tabs(selected="scan") as tabs:
            with _gr.Tab("Scan", id="scan"):
                with _gr.Row(equal_height=False):
                    with _gr.Column(scale=5, min_width=300):
                        img = _gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Receipt photo", height=380)
                        scan_btn = _gr.Button("Scan receipt", variant="primary", size="lg")
                        _gr.Examples([[p[0]] for p in example_paths], inputs=img, label="Or try a sample receipt (never used in training)")
                    with _gr.Column(scale=6, min_width=300):
                        card_html = _gr.HTML(EMPTY)
                        vendor_html = _gr.HTML()
                        with _gr.Group():
                            with _gr.Row():
                                company = _gr.Textbox(label="Merchant", scale=3)
                                total = _gr.Textbox(label="Total", scale=2)
                            with _gr.Row():
                                date = _gr.Textbox(label="Date", scale=2)
                                address = _gr.Textbox(label="Address", scale=3)
                        with _gr.Row():
                            add_btn = _gr.Button("Add to expenses", variant="primary")
                            to_split = _gr.Button("Split this bill", variant="secondary")
                with _gr.Accordion("Compare with the plain Gemma 4 model", open=False):
                    cmp_btn = _gr.Button("Run the comparison", size="sm")
                    cmp_html = _gr.HTML()
                    with _gr.Row():
                        ft_raw = _gr.Textbox(label="Fine-tuned (raw)", lines=4)
                        plain_raw = _gr.Textbox(label="Plain (raw)", lines=4)
            with _gr.Tab("Split", id="split"):
                with _gr.Row(equal_height=False):
                    with _gr.Column(scale=5, min_width=300):
                        with _gr.Group():
                            people = _gr.Textbox(label="Who's splitting?", value="Me, Riya, Arjun")
                            with _gr.Row():
                                payer = _gr.Textbox(label="Paid by", value="Me")
                                mode = _gr.Radio(["Equally", "By items"], value="Equally", label="Split")
                        items = _gr.Dataframe(headers=["item", "price", "shared by"], interactive=True, wrap=True,
                                              label="Items ('shared by': names or 'all')")
                        with _gr.Row():
                            fill_btn = _gr.Button("Auto-fill items", size="sm")
                            calc_btn = _gr.Button("Calculate split", variant="primary")
                    with _gr.Column(scale=6, min_width=300):
                        split_out = _gr.HTML('<div class="pc-card pc-empty-sm">Scan a receipt, then calculate the split here.</div>')
                        with _gr.Row():
                            group_btn = _gr.Button("Add to group", variant="primary")
                            reset_btn = _gr.Button("Reset group", size="sm")
                        group_out = _gr.HTML()
            with _gr.Tab("Expenses", id="expenses"):
                kpis = _gr.HTML()
                table = _gr.Dataframe(headers=["merchant", "date", "total", "address"], interactive=False, wrap=True, label="Expenses")
                with _gr.Row():
                    undo_btn = _gr.Button("Undo last", size="sm")
                    dl_btn = _gr.DownloadButton("Download CSV", interactive=False, size="sm")
        _gr.HTML(f'<div class="pc-foot">Open model: Gemma 4 E2B + one LoRA adapter file &middot; no cloud OCR, no subscription &middot; '
                 f'<a href="{HF}" target="_blank" rel="noopener">model card</a> &middot; photos are not stored or used for training</div>')

        scan_btn.click(lambda: ("", ""), outputs=[vendor_html, cmp_html], api_visibility="private").then(
            scan, inputs=img, outputs=[card_html, company, date, total, address, ft_raw, plain_raw], api_name="scan",
            concurrency_id="gpu", concurrency_limit=1).success(vendor, inputs=[company, address], outputs=vendor_html, api_visibility="private")
        cmp_btn.click(compare, inputs=[img, company, date, total, address], outputs=[cmp_html, plain_raw], api_name="compare",
                      concurrency_id="gpu", concurrency_limit=1)
        add_btn.click(add_exp, inputs=[company, date, total, address], outputs=[kpis, table, dl_btn], api_name="add_expense")
        undo_btn.click(undo_exp, outputs=[kpis, table, dl_btn], api_visibility="private")
        to_split.click(lambda: _gr.Tabs(selected="split"), outputs=tabs, api_visibility="private")
        fill_btn.click(autofill, inputs=img, outputs=[items, mode], api_name="autofill_items", concurrency_id="gpu", concurrency_limit=1)
        calc_btn.click(split, inputs=[people, payer, mode, items, total, company], outputs=[split_out, split_state], api_name="calc_split")
        group_btn.click(add_group, inputs=split_state, outputs=group_out, api_name="add_to_group")
        reset_btn.click(reset_g, outputs=group_out, api_visibility="private")
        app.load(exp_view, outputs=[kpis, table, dl_btn], api_visibility="private")
        app.load(group_html, outputs=group_out, api_visibility="private")
    app.queue(default_concurrency_limit=4, max_size=32)
    return app, theme, css, fav

# --- LAUNCH: a new public link for the polished app (the old link stays live)
pro_demo, _pro_theme, _pro_css, _pro_fav = _build_pro()
PRO_URL = pro_demo.launch(share=True, prevent_thread_lock=True, server_port=7863, quiet=True,
                          theme=_pro_theme, css=_pro_css, favicon_path=_pro_fav, allowed_paths=[OUT])[2]
print("\nPOLISHED APP (send this link to Claude):", PRO_URL)
