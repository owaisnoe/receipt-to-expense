# CELL: 15a. Receipt to Expense app + Split the bill: helpers and UI (about 5 s to build; no model call until someone presses Extract)
# Uses the globals from sections 0-12: model (PEFT-wrapped), generate_from_image, parse_json, norm_total, NORMALIZERS,
# FIELDS, OUT, test_subset. Both models get the same INSTRUCTION, greedy decoding at batch 1 and the 560-token budget
# through generate_from_image -> _encode, exactly as in the evaluation.
# Optional extras, used only if an earlier cell defined them:
#   extract_line_items(image, use_adapter=False) -> (items, raw, seconds)                 zero-shot line items (bonus)
#   vendor_lookup(company, address) -> dict(company, address, similarity, matched) | None  retrieval over TRAIN vendors (RAG bonus)
# CHANGED ON RUNTIME: added "Split the bill with friends" (student use case). Layout ideas from Mobbin: GoPay "Split bill
# details" (scanned items + an "Others" row for tax/service with "make sure this amount is correct"), Origin / Quicken
# ("Equal" vs specific split, "Left to split 0.00"), Brex "Itemize" (allocate by item).
import contextlib, hashlib, html, threading
import gradio as gr
from PIL import Image

FastVisionModel.for_inference(model)
os.makedirs(f"{OUT}/examples", exist_ok=True)
example_paths, EXAMPLE_GOLD = [], {}
for i, s in enumerate(test_subset[:3]):
    p = f"{OUT}/examples/receipt_{i}.jpg"; s["image"].convert("RGB").save(p); example_paths.append([p])
    _im = Image.open(p).convert("RGB")
    EXAMPLE_GOLD[(_im.size, hashlib.md5(_im.tobytes()).hexdigest())] = {k: s["entities"][k] for k in FIELDS}
if "expenses" not in globals():
    expenses = []                 # shared by everyone on the link, so a judge's receipt shows up in the presenter's table
if "GROUP" not in globals():
    GROUP = {"balances": {}, "history": []}   # split-the-bill group ledger, shared like the expenses table
_GPU = globals().get("GEN_LOCK") or threading.RLock()   # CHANGED ON RUNTIME: share the notebook-wide lock (section 14a)
_esc = lambda v: html.escape(str(v))

def _s(v):
    return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).strip()

def _same(k, a, b):
    return NORMALIZERS[k](a) != "" and NORMALIZERS[k](a) == NORMALIZERS[k](b)

def _gold_for(image):
    im = image.convert("RGB")
    if not any(im.size == size for size, _ in EXAMPLE_GOLD):
        return None
    return EXAMPLE_GOLD.get((im.size, hashlib.md5(im.tobytes()).hexdigest()))

def _generate(image, adapter):
    with _GPU:
        t = time.time()
        ctx = contextlib.nullcontext() if adapter or not hasattr(model, "disable_adapter") else model.disable_adapter()
        with ctx:
            raw = generate_from_image(image)          # INSTRUCTION, greedy, batch 1, MAX_SOFT_TOKENS via _encode
        return raw, time.time() - t

def _compare_html(f, pl, gold, t_ft, t_plain):
    chips = [f'<span class="r2e-chip">Fine-tuned &middot; {t_ft:.1f} s</span>',
             f'<span class="r2e-chip">Plain &middot; {"failed" if t_plain is None else f"{t_plain:.1f} s"}</span>']
    agree = sum(_same(k, _s(f.get(k)), _s(pl.get(k))) for k in FIELDS)
    chips.append(f'<span class="r2e-chip">Agree on {agree}/{len(FIELDS)} fields</span>')
    if gold:
        chips.append('<span class="r2e-chip">SROIE test receipt: scored against its label (normalized)</span>')
    mark = lambda ok, yes="&#10003;", no="&#10007;": f'<span class="r2e-yes">{yes}</span>' if ok else f'<span class="r2e-no">{no}</span>'
    head = "<tr><th>Field</th><th>Fine-tuned</th><th>Plain</th>" + ("<th>SROIE label</th>" if gold else "") + "<th>Fine-tuned vs plain</th></tr>"
    rows = []
    for k in FIELDS:
        a, b = _s(f.get(k)), _s(pl.get(k))
        if gold:
            g = _s(gold.get(k))
            cells = f"<td>{mark(_same(k, a, g))} {_esc(a)}</td><td>{mark(_same(k, b, g))} {_esc(b)}</td><td>{_esc(g)}</td>"
        else:
            cells = f"<td>{_esc(a)}</td><td>{_esc(b)}</td>"
        rows.append(f"<tr><th>{k.title()}</th>{cells}<td>{mark(_same(k, a, b), '&#10003; match', '&#10007; differs')}</td></tr>")
    return (f'<div class="r2e-h">Fine-tuned vs plain, field by field</div><div class="r2e-chips">{"".join(chips)}</div>'
            f'<div class="r2e-scroll"><table class="r2e-table">{head}{"".join(rows)}</table></div>')

def _trust_note(f, pl):
    """Fine-tuned vs plain agreement as a trust hint, quoting the rates measured on the 100 test receipts (section 13c)."""
    diff = [k for k in FIELDS if not _same(k, _s(f.get(k)), _s(pl.get(k)))]
    st, rate = globals().get("AGREE_STATS"), ""
    if st is not None and len(st):
        col = st["fine-tuned right when they agree (%)"]
        rate = f" On our test receipts, fields where both models agreed were right {col.min():.0f}-{col.max():.0f}% of the time."
    return (f'<div class="r2e-sub">Trust check: fine-tuned and plain agree on <b>{len(FIELDS) - len(diff)}/{len(FIELDS)}</b> fields'
            + (f"; double-check <b>{', '.join(diff)}</b>" if diff else "") + f".{rate}</div>")

def _callout_html(f, pl=None):
    flags = [n for n, ok in (("total", bool(norm_total(f.get("total", "")))), ("date", bool(re.search(r"\d", _s(f.get("date"))))),
                             ("company", bool(_s(f.get("company")))), ("address", bool(_s(f.get("address"))))) if not ok]
    trust = _trust_note(f, pl) if pl is not None else ""
    if flags:
        return f'<div class="r2e-callout warn">&#9888; Please check: <b>{", ".join(flags)}</b>. Fix the field before adding it.</div>' + trust
    return '<div class="r2e-callout ok">&#10003; All four fields parsed. Review them, then add to expenses or split the bill.</div>' + trust

def extract(image):
    if image is None:
        raise gr.Error("Upload or photograph a receipt first.")
    try:
        ft_raw, t_ft = _generate(image, adapter=True)
    except Exception as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"The fine-tuned model failed on this image ({type(e).__name__}: {str(e)[:200]}). Try another photo.")
    if not hasattr(model, "disable_adapter"):
        gr.Warning("No LoRA adapter is loaded, so both columns show the plain model.")
    try:
        plain_raw, t_plain = _generate(image, adapter=False)
    except Exception as e:
        torch.cuda.empty_cache()
        plain_raw, t_plain = f"(plain model failed: {type(e).__name__}: {str(e)[:200]})", None
        gr.Warning("The plain model failed on this image; the fine-tuned answer is still shown.")
    f, pl = parse_json(ft_raw), parse_json(plain_raw)
    if f is None:
        gr.Warning("The fine-tuned output was not valid JSON, so the fields are blank. See the raw output below.")
    f, pl = f or {}, pl or {}
    return (*[_s(f.get(k)) for k in FIELDS], ft_raw, plain_raw, _callout_html(f, pl if t_plain is not None else None),
            _compare_html(f, pl, _gold_for(image), t_ft, t_plain))

def reset_outputs(image):
    hidden = gr.HTML(value="", visible=False)
    return "", "", "", "", "", "", "", "", hidden, None, gr.Button(visible=False), ""

def lookup_vendor(company, address):
    fn = globals().get("vendor_lookup")
    hide = (gr.HTML(value="", visible=False), None, gr.Button(visible=False))
    if not callable(fn) or not (company or "").strip():
        return hide
    try:
        v = fn(company, address)
    except Exception as e:
        gr.Warning(f"Vendor lookup failed ({type(e).__name__}); the extracted fields are unchanged.")
        return hide
    if not v:
        return gr.HTML(value='<span class="r2e-badge muted">New vendor</span> not in the vendor index (training receipts only).',
                       visible=True), None, gr.Button(visible=False)
    sim = v.get("similarity")
    sim_txt = f" &middot; similarity {sim:.2f}" if isinstance(sim, (int, float)) else ""
    if not v.get("matched"):
        return gr.HTML(value=f'<span class="r2e-badge muted">New vendor</span> closest known: {_esc(v.get("company", ""))}{sim_txt}',
                       visible=True), None, gr.Button(visible=False)
    differs = not (_same("company", _s(v.get("company")), company) and _same("address", _s(v.get("address")), address))
    body = (f'<span class="r2e-badge ok">&#10003; Known vendor</span> <b>{_esc(v.get("company", ""))}</b>{sim_txt}'
            + (f'<div class="r2e-sub">Training-set address: {_esc(v.get("address", ""))}</div>' if differs else ""))
    return gr.HTML(value=body, visible=True), v, gr.Button(visible=differs)

def apply_vendor(v, company, address):
    if not v:
        return company, address
    gr.Info("Used the known vendor's name and address. Check them before adding.")
    return _s(v.get("company")) or company, _s(v.get("address")) or address

def run_line_items(image, total):
    fn = globals().get("extract_line_items")
    if not callable(fn):
        raise gr.Error("Line-item extraction is not loaded in this notebook.")
    if image is None:
        raise gr.Error("Upload or photograph a receipt first.")
    try:
        with _GPU:
            items, raw, secs = fn(image, use_adapter=False)
    except Exception as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"Line-item extraction failed ({type(e).__name__}: {str(e)[:200]}).")
    cols = ["name", "quantity", "price"]
    df = pd.DataFrame([{c: _s(it.get(c)) for c in cols} for it in (items or []) if isinstance(it, dict)], columns=cols)
    prices = [float(norm_total(x)) for x in df["price"] if norm_total(x)]
    check = ""
    if prices and norm_total(total):
        ok = abs(sum(prices) - float(norm_total(total))) < 0.01
        check = (f' &middot; sum of prices {sum(prices):.2f} vs total {norm_total(total)} '
                 + ('<span class="r2e-yes">&#10003;</span>' if ok else '<span class="r2e-no">&#8800;</span> (prices may be unit prices, or tax and rounding are separate)'))
    summary = (f'<div class="r2e-h">Line items ({len(df)})</div><div class="r2e-sub">Zero-shot, plain Gemma 4 '
               f'(SROIE has no line-item labels, so this is not scored) &middot; {secs:.1f} s{check}</div>')
    return df, summary, raw

def _render_expenses():
    df = pd.DataFrame(expenses, columns=["company", "date", "address", "total"])
    spend = sum(float(norm_total(x) or 0) for x in df["total"])
    path = f"{OUT}/expenses.csv"; df.to_csv(path, index=False)
    last = f'{_esc(df.iloc[-1]["company"] or "(no name)")} &middot; {_esc(df.iloc[-1]["total"])}' if len(df) else "&mdash;"
    kpis = (f'<div class="r2e-kpis"><div class="r2e-kpi"><div class="l">Total spend</div><div class="v">{spend:.2f}</div></div>'
            f'<div class="r2e-kpi"><div class="l">Receipts logged</div><div class="v">{len(df)}</div></div>'
            f'<div class="r2e-kpi"><div class="l">Last added</div><div class="v small">{last}</div></div></div>')
    return df, kpis, (path if len(df) else None)

def add_expense(company, date, address, total):
    if not any(_s(x) for x in (company, date, address, total)):
        raise gr.Error("Nothing to add yet: extract a receipt or type the fields first.")
    expenses.append({"company": _s(company), "date": _s(date), "address": _s(address), "total": _s(total)})
    gr.Info(f"Added {_s(company) or 'expense'} ({_s(total) or 'no total'}).")
    return _render_expenses()

def undo_last():
    if expenses:
        expenses.pop()
    return _render_expenses()

# ---- Split the bill with friends -------------------------------------------------------------------------------
def _names(s):
    out = []
    for n in re.split(r"[,;\n]+", str(s or "")):
        n = n.strip()
        if n and n.lower() not in {x.lower() for x in out}:
            out.append(n)
    return out

def _amt(x):
    v = norm_total(_s(x))
    return float(v) if v else None

def _money(v):
    return f"{v:,.2f}"

def _rows(df):
    if df is None:
        return []
    if isinstance(df, dict):                       # {"headers": [...], "data": [[...]]} from the API
        df = pd.DataFrame(df.get("data") or [], columns=(df.get("headers") or None))
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return [[_s(x) for x in r] for r in df.itertuples(index=False)]

def fill_split_items(image, li_df):
    """Items for the split: the line items already extracted, else run the zero-shot line-item extractor now."""
    rows = [[r[0], r[-1], "all"] for r in _rows(li_df) if r and (r[0] or r[-1])]
    if not rows:
        fn = globals().get("extract_line_items")
        if image is None:
            raise gr.Error("Upload or photograph a receipt first.")
        if not callable(fn):
            raise gr.Error("Line-item extraction is not loaded: type the items by hand, or split equally.")
        try:
            with _GPU:
                items, raw, secs = fn(image, use_adapter=False)
        except Exception as e:
            torch.cuda.empty_cache()
            raise gr.Error(f"Line-item extraction failed ({type(e).__name__}).")
        rows = [[_s(it.get("name")), _s(it.get("price")), "all"] for it in (items or []) if isinstance(it, dict)]
    if not rows:
        gr.Warning("No line items found. Add rows by hand, or split equally.")
    return pd.DataFrame(rows, columns=["item", "price", "shared by"]), gr.Radio(value="By items")

def calc_split(people, payer, mode, items_df, total, company):
    """Shares per person. 'By items': each item is split among the people named in 'shared by' ('all' = everyone);
    the rest of the total (tax, service, rounding, discount = 'Others') is shared in proportion to the items."""
    names = _names(people)
    if len(names) < 2:
        raise gr.Error("Add at least two people, separated by commas.")
    payer_n = next((n for n in names if n.lower() == _s(payer).lower()), None)
    if payer_n is None:
        raise gr.Error(f"'Paid by' must be one of: {', '.join(names)}.")
    tot = _amt(total)
    if tot is None or tot <= 0:
        raise gr.Error("Extract a receipt (or type the total) first.")
    shares, items_of, notes, others, items_sum = {n: 0.0 for n in names}, {n: [] for n in names}, [], 0.0, None
    if mode == "By items":
        priced = 0
        for r in _rows(items_df):
            r = r + ["", "", ""]
            item, price, who = r[0], _amt(r[1]), r[2]
            if price is None:
                if item:
                    notes.append(f"skipped '{item}' (no price)")
                continue
            wanted = [w.strip().lower() for w in re.split(r"[,;/&+]|\band\b", who) if w.strip()]
            ppl = names if not wanted or wanted[0] in ("all", "everyone", "everybody") else [n for n in names if n.lower() in wanted]
            if not ppl:
                notes.append(f"'{item}': nobody called '{who}', so it is shared by everyone")
                ppl = names
            for n in ppl:
                shares[n] += price / len(ppl)
                items_of[n].append(item or "item")
            priced += 1
        if not priced:
            raise gr.Error("No priced items: press 'Fill items from receipt', type them, or switch to 'Equally'.")
        items_sum = sum(shares.values())
        others = round(tot - items_sum, 2)
        if abs(others) >= 0.01:
            for n in names:
                shares[n] += others * (shares[n] / items_sum) if items_sum else others / len(names)
    else:
        for n in names:
            shares[n], items_of[n] = tot / len(names), ["equal share"]
    rounded = {n: round(v, 2) for n, v in shares.items()}
    rounded[payer_n] = round(rounded[payer_n] + (tot - sum(rounded.values())), 2)   # the cent remainder stays with the payer
    left = round(tot - sum(rounded.values()), 2)
    chips = [f'<span class="r2e-chip">Total {_money(tot)}</span>', f'<span class="r2e-chip">Paid by {_esc(payer_n)}</span>',
             f'<span class="r2e-chip">{len(names)} people &middot; {"by items" if mode == "By items" else "equally"}</span>']
    warn = ""
    if items_sum is not None:
        chips.append(f'<span class="r2e-chip">Items {_money(items_sum)}</span>')
        if abs(others) >= 0.01:
            big = abs(others) > 0.25 * tot
            warn = (f'<div class="r2e-callout {"warn" if big else "ok"}">{"&#9888; " if big else ""}Others (tax, service, rounding'
                    f'{", discount" if others < 0 else ""}): <b>{_money(others)}</b>, shared in proportion to each person\'s items.'
                    + (" Make sure this amount is correct: item prices may be unit prices, or an item is missing." if big else "")
                    + "</div>")
    tick = '<span class="r2e-yes">&#10003;</span>' if abs(left) < 0.005 else ""
    body = "".join(
        f'<tr><th>{_esc(n)}</th><td>{_esc(", ".join(items_of[n][:6]) + (" ..." if len(items_of[n]) > 6 else ""))}</td>'
        f'<td><b>{_money(rounded[n])}</b></td>'
        f'<td>{"paid the bill" if n == payer_n else "owes " + _esc(payer_n) + " <b>" + _money(rounded[n]) + "</b>"}</td></tr>'
        for n in names)
    out_html = (f'<div class="r2e-h">Split: {_esc(_s(company)) or "this receipt"}</div><div class="r2e-chips">{"".join(chips)}</div>{warn}'
                f'<div class="r2e-scroll"><table class="r2e-table"><tr><th>Person</th><th>Items</th><th>Share</th><th>Settle</th></tr>'
                f'{body}</table></div><div class="r2e-sub">Left to split: <b>{_money(left)}</b> {tick}</div>'
                + (f'<div class="r2e-sub">{_esc("; ".join(notes))}</div>' if notes else ""))
    return out_html, {"label": _s(company) or "receipt", "payer": payer_n, "total": tot, "shares": rounded}

def _settle(bal):
    """Fewest-payments settle-up: the largest debtor pays the largest creditor, repeatedly."""
    cred = sorted([[n, v] for n, v in bal.items() if v > 0.005], key=lambda x: -x[1])
    debt = sorted([[n, -v] for n, v in bal.items() if v < -0.005], key=lambda x: -x[1])
    out, i, j = [], 0, 0
    while i < len(debt) and j < len(cred):
        amt = round(min(debt[i][1], cred[j][1]), 2)
        if amt >= 0.01:
            out.append((debt[i][0], cred[j][0], amt))
        debt[i][1] -= amt; cred[j][1] -= amt
        if debt[i][1] < 0.005: i += 1
        if cred[j][1] < 0.005: j += 1
    return out

def _render_group():
    bal, hist = GROUP["balances"], GROUP["history"]
    if not hist:
        return '<div class="r2e-sub">No shared receipts yet: calculate a split, then add it to the group.</div>', None
    kp = "".join(f'<div class="r2e-kpi"><div class="l">{_esc(n)}</div><div class="v">{"+" if v > 0 else ""}{_money(v)}</div></div>'
                 for n, v in sorted(bal.items(), key=lambda x: -x[1]))
    moves = "".join(f"<li><b>{_esc(a)}</b> pays <b>{_esc(b)}</b> {_money(x)}</li>" for a, b, x in _settle(bal)) \
        or "<li>All settled up &#10003;</li>"
    path = f"{OUT}/group_splits.csv"; pd.DataFrame(hist).to_csv(path, index=False)
    return (f'<div class="r2e-h">Group balances ({len(hist)} receipt{"s" if len(hist) != 1 else ""})</div>'
            '<div class="r2e-sub">+ = the group owes them, - = they owe the group.</div>'
            f'<div class="r2e-kpis">{kp}</div><div class="r2e-h">Settle up with the fewest payments</div><ul>{moves}</ul>'), path

def add_to_group(state):
    if not state:
        raise gr.Error("Calculate a split first.")
    bal = GROUP["balances"]
    for n, sh in state["shares"].items():
        bal[n] = round(bal.get(n, 0.0) - sh, 2)
    bal[state["payer"]] = round(bal.get(state["payer"], 0.0) + state["total"], 2)
    GROUP["history"].append({"receipt": state["label"], "paid_by": state["payer"], "total": state["total"],
                             **{f"share {n}": v for n, v in state["shares"].items()}})
    gr.Info(f"Added {state['label']} to the group balances.")
    return _render_group()

def reset_group():
    GROUP["balances"].clear(); GROUP["history"].clear()
    return _render_group()

HAS_LINE_ITEMS = callable(globals().get("extract_line_items"))
_hf = globals().get("HF_REPO") or ""
_hf_link = (f' &middot; <a href="https://huggingface.co/{_esc(_hf.strip())}" target="_blank" rel="noopener">Model card on Hugging Face</a>'
            if isinstance(_hf, str) and _hf.strip() else "")
_setup = (f'Gemma 4 E2B + LoRA (r={globals().get("LORA_R", "?")}) fine-tuned on the 626 SROIE training receipts &middot; '
          f'visual budget {globals().get("MAX_SOFT_TOKENS", "?")} tokens &middot; greedy decoding')

APP_THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")
APP_CSS = """
.r2e-hero h1 {margin: 0; font-size: 1.7rem; line-height: 1.2}
.r2e-hero p {margin: .3rem 0 .6rem; color: var(--body-text-color-subdued)}
.r2e-steps {display: flex; gap: .5rem; flex-wrap: wrap}
.r2e-step {padding: .2rem .7rem; border-radius: 999px; font-size: .85rem;
           background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary)}
.r2e-h {font-weight: 700; font-size: 1.05rem; margin: .2rem 0}
.r2e-sub {color: var(--body-text-color-subdued); font-size: .9rem; margin: .15rem 0}
.r2e-callout {border-radius: 10px; padding: .55rem .8rem; font-size: .92rem; margin: .2rem 0}
.r2e-callout.warn {background: rgba(245, 158, 11, .14); border: 1px solid rgba(245, 158, 11, .6)}
.r2e-callout.ok {background: rgba(16, 185, 129, .12); border: 1px solid rgba(16, 185, 129, .5)}
.r2e-badge {display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: .3rem}
.r2e-badge.ok {background: rgba(16, 185, 129, .16); border: 1px solid rgba(16, 185, 129, .55)}
.r2e-badge.muted {background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary)}
.r2e-chips {display: flex; gap: .4rem; flex-wrap: wrap; margin: .35rem 0 .5rem}
.r2e-chip {padding: .15rem .6rem; border-radius: 999px; font-size: .82rem; background: var(--background-fill-secondary);
           border: 1px solid var(--border-color-primary)}
.r2e-scroll {overflow-x: auto}
.r2e-table {width: 100%; border-collapse: collapse; font-size: .9rem}
.r2e-table th, .r2e-table td {padding: .45rem .5rem; border-bottom: 1px solid var(--border-color-primary); text-align: left;
                              vertical-align: top; overflow-wrap: anywhere}
.r2e-yes {color: #10b981; font-weight: 700}
.r2e-no {color: #ef4444; font-weight: 700}
.r2e-kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .6rem; margin: .3rem 0}
.r2e-kpi {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .6rem .8rem; background: var(--block-background-fill)}
.r2e-kpi .l {font-size: .75rem; letter-spacing: .05em; text-transform: uppercase; color: var(--body-text-color-subdued)}
.r2e-kpi .v {font-size: 1.45rem; font-weight: 700; margin-top: .1rem}
.r2e-kpi .v.small {font-size: 1rem; overflow-wrap: anywhere}
.r2e-footer {font-size: .85rem; color: var(--body-text-color-subdued); text-align: center; padding: .8rem 0 .2rem}
"""

gr.close_all()
with gr.Blocks(title="Receipt to Expense") as demo:
    gr.HTML(f'<div class="r2e-hero"><h1>Receipt to Expense</h1><p>{_setup}</p><div class="r2e-steps">'
            '<span class="r2e-step">1 &middot; Snap or upload a receipt</span><span class="r2e-step">2 &middot; Review the fields</span>'
            '<span class="r2e-step">3 &middot; Add to your expenses</span><span class="r2e-step">4 &middot; Split with friends</span></div></div>')
    vendor_state = gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=300):
            img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Receipt", height=440)
            btn = gr.Button("Extract", variant="primary", size="lg")
            gr.Examples(example_paths, inputs=img, label="Try a receipt from the SROIE test split (never trained on)")
        with gr.Column(scale=6, min_width=300):
            gr.HTML('<div class="r2e-h">Review before adding</div>'
                    '<div class="r2e-sub">The fine-tuned model fills these in. Edit anything that looks wrong.</div>')
            callout = gr.HTML()
            vendor_html = gr.HTML(visible=False)
            apply_btn = gr.Button("Use the known vendor's name and address", size="sm", visible=False)
            company = gr.Textbox(label="Company")
            address = gr.Textbox(label="Address", lines=2)
            with gr.Row():
                date = gr.Textbox(label="Date", min_width=140)
                total = gr.Textbox(label="Total", min_width=140)
            add_btn = gr.Button("Add to expenses", variant="primary")
    compare = gr.HTML()
    with gr.Accordion("Raw model outputs", open=False):
        with gr.Row():
            ft_box = gr.Textbox(label="Fine-tuned Gemma (raw output)", lines=6)
            plain_box = gr.Textbox(label="Plain Gemma (raw output)", lines=6)
    with gr.Accordion("Line items (zero-shot extra)", open=False, visible=HAS_LINE_ITEMS):
        li_btn = gr.Button("Extract line items")
        li_summary = gr.HTML()
        li_table = gr.Dataframe(headers=["name", "quantity", "price"], interactive=False, wrap=True, label="Line items")
        li_raw = gr.Textbox(label="Raw output", lines=4)
    with gr.Accordion("Split the bill with friends", open=True):
        gr.HTML('<div class="r2e-sub">Shared a meal or a trip? Split this receipt equally or item by item. Tax, service and '
                'rounding ("Others") are shared in proportion, and the group keeps a running "who owes whom".</div>')
        with gr.Row():
            people = gr.Textbox(label="Who is splitting? (comma-separated)", value="Me, Riya, Arjun", scale=3)
            payer = gr.Textbox(label="Paid by", value="Me", scale=1)
            split_mode = gr.Radio(["Equally", "By items"], value="Equally", label="Split", scale=2)
        split_items = gr.Dataframe(headers=["item", "price", "shared by"], interactive=True, wrap=True,
                                   label="Items: put names in 'shared by' (for example: Riya, Arjun), or 'all'")
        with gr.Row():
            fill_btn = gr.Button("Fill items from receipt", size="sm")
            split_btn = gr.Button("Calculate split", variant="primary")
            group_btn = gr.Button("Add to group balances")
            reset_group_btn = gr.Button("Reset group", size="sm")
        split_html = gr.HTML()
        split_state = gr.State(None)
        group_html = gr.HTML()
        group_csv = gr.File(label="Group splits CSV")
    gr.HTML('<div class="r2e-h" style="margin-top:.6rem">Expenses</div>')
    kpis = gr.HTML()
    table = gr.Dataframe(headers=["company", "date", "address", "total"], interactive=False, wrap=True, label="Expenses")
    with gr.Row():
        undo_btn = gr.Button("Undo last", size="sm")
        csv_out = gr.File(label="Download CSV")
    gr.HTML(f'<div class="r2e-footer">Open model: Gemma 4 E2B (2B-class) + one LoRA adapter file, so no cloud OCR and no subscription. '
            f'Uploaded images are not saved or used for training (Gradio\'s temporary upload cache '
            f'stays on this Colab runtime and is wiped when the runtime is deleted).{_hf_link}</div>')

    img.change(reset_outputs, inputs=img, outputs=[company, date, address, total, ft_box, plain_box, callout, compare,
                                                   vendor_html, vendor_state, apply_btn, split_html], api_visibility="private")
    btn.click(extract, inputs=img, outputs=[company, date, address, total, ft_box, plain_box, callout, compare],
              api_name="extract", concurrency_id="gpu", concurrency_limit=1
              ).success(lookup_vendor, inputs=[company, address], outputs=[vendor_html, vendor_state, apply_btn], api_name="lookup_vendor")
    apply_btn.click(apply_vendor, inputs=[vendor_state, company, address], outputs=[company, address], api_visibility="private")
    add_btn.click(add_expense, inputs=[company, date, address, total], outputs=[table, kpis, csv_out], api_name="add_expense")
    undo_btn.click(undo_last, outputs=[table, kpis, csv_out], api_visibility="private")
    li_btn.click(run_line_items, inputs=[img, total], outputs=[li_table, li_summary, li_raw], api_name="line_items",
                 concurrency_id="gpu", concurrency_limit=1)
    fill_btn.click(fill_split_items, inputs=[img, li_table], outputs=[split_items, split_mode], api_name="fill_split_items",
                   concurrency_id="gpu", concurrency_limit=1)
    split_btn.click(calc_split, inputs=[people, payer, split_mode, split_items, total, company],
                    outputs=[split_html, split_state], api_name="calc_split")
    group_btn.click(add_to_group, inputs=split_state, outputs=[group_html, group_csv], api_name="add_to_group")
    reset_group_btn.click(reset_group, outputs=[group_html, group_csv], api_name="reset_group")
    demo.load(_render_expenses, outputs=[table, kpis, csv_out], api_visibility="private")
    demo.load(_render_group, outputs=[group_html, group_csv], api_visibility="private")
demo.queue(default_concurrency_limit=1, max_size=32)
print("App built. Line items panel:", "on" if HAS_LINE_ITEMS else "off (extract_line_items not defined)",
      "| vendor badge:", "on" if callable(globals().get("vendor_lookup")) else "off (vendor_lookup not defined)",
      "| split the bill: on | model card link:", "on" if _hf_link else "off (HF_REPO empty)")
