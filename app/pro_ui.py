"""Receipt to Expense - polished frontend (runs on the laptop). The fine-tuned Gemma 4 E2B runs on the Colab GPU behind
the existing Gradio app; this UI calls its /extract and /fill_split_items API. Split, group balances and expenses run here.
Layout ideas from Mobbin: Navan / Expensify expense card, GoPay split bill ("Others" + "make sure"), Origin / Quicken."""
import html, json, os, re, threading, time
import gradio as gr
import pandas as pd
from gradio_client import Client, handle_file
from PIL import Image, ImageOps

BACKEND = os.environ.get("R2E_BACKEND", "https://f6753b53a276c78344.gradio.live")
HF_URL = "https://huggingface.co/owaisnoe/gemma4-e2b-sroie-lora"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
EXAMPLES = [[os.path.join(HERE, "examples", f"receipt_{i}.jpg")] for i in range(3)]
_client, _lock = None, threading.Lock()
expenses, GROUP = [], {"balances": {}, "history": []}
_esc = lambda v: html.escape(str(v))

def _cl():
    global _client
    if _client is None:
        _client = Client(BACKEND, verbose=False)
    return _client

FAST_FILE, _fast = os.path.join(HERE, "fast_url.txt"), {"url": None, "client": None}

def _fast_client():
    """Fine-tuned-only fast endpoint (PASTE_INTO_COLAB.py), if its link was saved in fast_url.txt; else None."""
    url = (open(FAST_FILE).read().strip() if os.path.exists(FAST_FILE) else os.environ.get("R2E_FAST", "")).rstrip("/")
    if not url:
        return None
    if url != _fast["url"]:
        _fast.update(url=url, client=Client(url, verbose=False))
    return _fast["client"]

def _prep(path):
    """Upright, at most 2048 px on the long side (the 1120-token budget never uses more), JPEG: a much faster upload."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((2048, 2048))
    out = os.path.join(OUT, f"upload_{int(time.time() * 1000)}.jpg")
    im.save(out, quality=90)
    return out

def _s(v):
    return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).strip()

def norm_total(s):
    nums = re.findall(r"\d+(?:\.\d+)?", str(s or "").replace(",", ""))
    return f"{float(nums[-1]):.2f}" if nums else ""

def _money(v):
    return f"{v:,.2f}"

# ---------------------------------------------------------------- scan
EMPTY_CARD = ('<div class="pc-card pc-empty"><div class="pc-empty-icon">&#129534;</div><div class="pc-empty-t">No receipt yet</div>'
              '<div class="pc-muted">Snap or upload a receipt, then press <b>Scan receipt</b>. The fine-tuned model reads '
              'the merchant, date, address and total.</div></div>')

def _summary_card(company, date, address, total, callout, secs, vendor_html):
    m = re.search(r"agree on <b>(\d)/(\d)</b> fields(?:; double-check <b>([^<]+)</b>)?", callout or "")
    flags = re.search(r"Please check: <b>([^<]+)</b>", callout or "")
    chips = []
    chips.append(f'<span class="pc-chip pc-warn">&#9888; Check {_esc(flags.group(1))}</span>' if flags
                 else '<span class="pc-chip pc-ok">&#10003; All 4 fields read</span>')
    if m:
        agree, n, diff = m.group(1), m.group(2), m.group(3)
        chips.append(f'<span class="pc-chip {"pc-ok" if not diff else "pc-warn"}">Models agree {agree}/{n}'
                     + (f" &middot; double-check {_esc(diff)}" if diff else "") + "</span>")
    if "Known vendor" in (vendor_html or ""):
        chips.append('<span class="pc-chip pc-ok">&#10003; Known vendor</span>')
    elif "New vendor" in (vendor_html or ""):
        chips.append('<span class="pc-chip">New vendor</span>')
    chips.append(f'<span class="pc-chip">{secs:.1f} s</span>')
    initial = (_s(company)[:1] or "?").upper()
    note = ("Fields agreed by both models were right 83-100% of the time on our 100 test receipts." if m else
            "Read by the fine-tuned model. Open the comparison below to see the plain model side by side.")
    return (f'<div class="pc-card pc-summary"><div class="pc-row"><div class="pc-avatar">{_esc(initial)}</div>'
            f'<div class="pc-grow"><div class="pc-merchant">{_esc(company) or "Unknown merchant"}</div>'
            f'<div class="pc-muted">{_esc(date) or "no date"}</div></div>'
            f'<div class="pc-amount"><span class="pc-cur">Total</span>{_esc(norm_total(total) or total or "-")}</div></div>'
            f'<div class="pc-address">{_esc(address)}</div><div class="pc-chips">{"".join(chips)}</div>'
            f'<div class="pc-note">{note}</div></div>')

def scan(image):
    if not image:
        raise gr.Error("Add a receipt photo first (upload, camera or an example).")
    t = time.time()
    path = _prep(image)
    try:
        fc = _fast_client()
    except Exception:
        fc = None
    try:
        with _lock:
            if fc is not None:     # fast path: fine-tuned model only
                company, date, address, total, ft_raw, _ = fc.predict(handle_file(path), api_name="/fast_extract")
                plain_raw = callout = compare = ""
            else:                  # full path: fine-tuned + plain (trust check)
                company, date, address, total, ft_raw, plain_raw, callout, compare = _cl().predict(handle_file(path), api_name="/extract")
    except Exception as e:
        raise gr.Error(f"The model server did not answer ({type(e).__name__}). Try again in a few seconds.")
    card = _summary_card(company, date, address, total, callout, time.time() - t, "")
    return card, company, date, total, address, compare, ft_raw, plain_raw, path

def vendor_badge(company, address):
    """Runs after the card is shown, so it never slows the scan."""
    if not _s(company):
        return ""
    try:
        with _lock:
            v = _cl().predict(company, address, api_name="/lookup_vendor")
        v0 = v[0] if isinstance(v, (list, tuple)) else v
        h = v0.get("value", "") if isinstance(v0, dict) else _s(v0)
    except Exception:
        return ""
    if "Known vendor" in h:
        return '<div class="pc-vendor pc-ok">&#10003; Known vendor: seen in the training receipts</div>'
    return '<div class="pc-vendor">New vendor: not in the training receipts</div>' if "New vendor" in h else ""

def compare_models(path):
    if not path:
        raise gr.Error("Scan a receipt first.")
    try:
        with _lock:
            r = _cl().predict(handle_file(path), api_name="/extract")
    except Exception as e:
        raise gr.Error(f"The comparison did not run ({type(e).__name__}).")
    m = re.search(r"agree on <b>(\d)/(\d)</b> fields(?:; double-check <b>([^<]+)</b>)?", r[6] or "")
    trust = (f'<div class="pc-banner">Fine-tuned and plain agree on <b>{m.group(1)}/{m.group(2)}</b> fields'
             + (f"; double-check <b>{_esc(m.group(3))}</b>" if m.group(3) else "") + ". Agreed fields were right 83-100% of the time on our test receipts.</div>") if m else ""
    return trust + (r[7] or ""), r[4], r[5]

# ---------------------------------------------------------------- split (same maths as the tested Colab app)
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

def _rows(df):
    if df is None:
        return []
    if isinstance(df, dict):
        df = pd.DataFrame(df.get("data") or [], columns=(df.get("headers") or None))
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return [[_s(x) for x in r] for r in df.itertuples(index=False)]

def autofill_items(image, path):
    src = path or (image and _prep(image))
    if not src:
        raise gr.Error("Scan a receipt first.")
    rows = []
    try:
        fc = _fast_client()
    except Exception:
        fc = None
    try:
        with _lock:
            if fc is not None:
                js, _ = fc.predict(handle_file(src), api_name="/fast_items")
                rows = [[_s(it.get("name")), _s(it.get("price"))] for it in json.loads(js or "[]") if isinstance(it, dict)]
            else:              # /line_items: zero-shot line items on the public app (name, quantity, price)
                df, _, _ = _cl().predict(handle_file(src), "", api_name="/line_items")
                rows = [[r[0], r[-1]] for r in ((df.get("data") if isinstance(df, dict) else []) or []) if r]
    except Exception as e:
        raise gr.Error(f"Line-item reading failed ({type(e).__name__}).")
    if not rows:
        gr.Warning("No line items found: add rows by hand, or split equally.")
    return pd.DataFrame([[n, p, "all"] for n, p in rows], columns=["item", "price", "shared by"]), gr.Radio(value="By items")

def calc_split(people, payer, mode, items_df, total, company):
    names = _names(people)
    if len(names) < 2:
        raise gr.Error("Add at least two people, separated by commas.")
    payer_n = next((n for n in names if n.lower() == _s(payer).lower()), None)
    if payer_n is None:
        raise gr.Error(f"'Paid by' must be one of: {', '.join(names)}.")
    tot = _amt(total)
    if tot is None or tot <= 0:
        raise gr.Error("Scan a receipt (or type the total) first.")
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
                shares[n] += price / len(ppl); items_of[n].append(item or "item")
            priced += 1
        if not priced:
            raise gr.Error("No priced items: press 'Auto-fill items', type them, or split equally.")
        items_sum = sum(shares.values())
        others = round(tot - items_sum, 2)
        if abs(others) >= 0.01:
            for n in names:
                shares[n] += others * (shares[n] / items_sum) if items_sum else others / len(names)
    else:
        for n in names:
            shares[n], items_of[n] = tot / len(names), ["equal share"]
    rounded = {n: round(v, 2) for n, v in shares.items()}
    rounded[payer_n] = round(rounded[payer_n] + (tot - sum(rounded.values())), 2)
    left = round(tot - sum(rounded.values()), 2)
    warn = ""
    if items_sum is not None and abs(others) >= 0.01:
        big = abs(others) > 0.25 * tot
        warn = (f'<div class="pc-banner {"pc-banner-warn" if big else ""}">Others (tax, service, rounding'
                f'{", discount" if others < 0 else ""}): <b>{_money(others)}</b>, shared in proportion.'
                + (" Make sure this amount is correct." if big else "") + "</div>")
    rows = "".join(
        f'<div class="pc-person"><div class="pc-avatar pc-sm">{_esc(n[:1].upper())}</div><div class="pc-grow">'
        f'<div class="pc-pname">{_esc(n)}</div><div class="pc-muted">{_esc(", ".join(items_of[n][:4]))}</div></div>'
        f'<div class="pc-right"><div class="pc-pamt">{_money(rounded[n])}</div>'
        f'<div class="{"pc-paid" if n == payer_n else "pc-owes"}">{"paid the bill" if n == payer_n else "owes " + _esc(payer_n)}</div></div></div>'
        for n in names)
    tick = "&#10003;" if abs(left) < 0.005 else ""
    out = (f'<div class="pc-card"><div class="pc-row"><div class="pc-grow"><div class="pc-merchant">{_esc(_s(company)) or "This receipt"}</div>'
           f'<div class="pc-muted">{len(names)} people &middot; {"by item" if mode == "By items" else "split equally"} &middot; paid by {_esc(payer_n)}</div></div>'
           f'<div class="pc-amount"><span class="pc-cur">Total</span>{_money(tot)}</div></div>{warn}{rows}'
           f'<div class="pc-left">Left to split <b>{_money(left)}</b> <span class="pc-tick">{tick}</span></div>'
           + (f'<div class="pc-muted">{_esc("; ".join(notes))}</div>' if notes else "") + "</div>")
    return out, {"label": _s(company) or "receipt", "payer": payer_n, "total": tot, "shares": rounded}

def _settle(bal):
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
        return '<div class="pc-card pc-empty-sm">No shared bills yet. Calculate a split, then add it to the group.</div>'
    tiles = "".join(f'<div class="pc-kpi"><div class="pc-kl">{_esc(n)}</div><div class="pc-kv {"pc-pos" if v > 0 else "pc-neg" if v < 0 else ""}">'
                    f'{"+" if v > 0 else ""}{_money(v)}</div></div>' for n, v in sorted(bal.items(), key=lambda x: -x[1]))
    moves = "".join(f'<div class="pc-move"><b>{_esc(a)}</b> pays <b>{_esc(b)}</b><span>{_money(x)}</span></div>'
                    for a, b, x in _settle(bal)) or '<div class="pc-move">All settled up &#10003;</div>'
    return (f'<div class="pc-card"><div class="pc-h">Group balances <span class="pc-muted">({len(hist)} bill{"s" if len(hist) != 1 else ""})</span></div>'
            f'<div class="pc-kpis">{tiles}</div><div class="pc-h">Settle up with the fewest payments</div>{moves}</div>')

def add_to_group(state):
    if not state:
        raise gr.Error("Calculate a split first.")
    bal = GROUP["balances"]
    for n, sh in state["shares"].items():
        bal[n] = round(bal.get(n, 0.0) - sh, 2)
    bal[state["payer"]] = round(bal.get(state["payer"], 0.0) + state["total"], 2)
    GROUP["history"].append({"bill": state["label"], "paid_by": state["payer"], "total": state["total"], **state["shares"]})
    gr.Info(f"Added {state['label']} to the group.")
    return _render_group()

def reset_group():
    GROUP["balances"].clear(); GROUP["history"].clear()
    return _render_group()

# ---------------------------------------------------------------- expenses
def _render_expenses():
    df = pd.DataFrame(expenses, columns=["merchant", "date", "total", "address"])
    spend = sum(float(norm_total(x) or 0) for x in df["total"])
    path = os.path.join(OUT, "expenses.csv"); df.to_csv(path, index=False)
    last = f'{_esc(df.iloc[-1]["merchant"] or "-")}' if len(df) else "-"
    kpis = (f'<div class="pc-kpis"><div class="pc-kpi"><div class="pc-kl">Total spend</div><div class="pc-kv">{_money(spend)}</div></div>'
            f'<div class="pc-kpi"><div class="pc-kl">Receipts</div><div class="pc-kv">{len(df)}</div></div>'
            f'<div class="pc-kpi"><div class="pc-kl">Last added</div><div class="pc-kv pc-small">{last}</div></div></div>')
    return kpis, df, gr.DownloadButton(value=path if len(df) else None, interactive=bool(len(df)))

def add_expense(company, date, total, address):
    if not any(_s(x) for x in (company, date, total, address)):
        raise gr.Error("Nothing to add yet: scan a receipt first.")
    expenses.append({"merchant": _s(company), "date": _s(date), "total": _s(total), "address": _s(address)})
    gr.Info(f"Added {_s(company) or 'expense'} ({_s(total)}).")
    return _render_expenses()

def undo_expense():
    if expenses:
        expenses.pop()
    return _render_expenses()

# ---------------------------------------------------------------- look
THEME = gr.themes.Base(primary_hue="emerald", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    body_background_fill="#f5f6f8", block_background_fill="#ffffff", block_border_color="#e6e8ec", block_border_width="1px",
    block_radius="16px", block_shadow="0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06)", block_label_text_weight="600",
    button_primary_background_fill="#0f9f6e", button_primary_background_fill_hover="#0b8a5f", button_primary_text_color="#ffffff",
    button_large_radius="12px", button_small_radius="10px", input_radius="10px", input_background_fill="#f8fafc")
CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important}
footer {display: none !important}
.pc-appbar {display:flex; align-items:center; gap:14px; padding:14px 6px 6px}
.pc-logo {width:44px; height:44px; border-radius:12px; background:linear-gradient(135deg,#10b981,#047857); color:#fff;
          display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 6px 16px rgba(16,185,129,.35)}
.pc-title {font-size:1.45rem; font-weight:700; letter-spacing:-.01em; color:#0f172a}
.pc-sub {color:#64748b; font-size:.92rem}
.pc-pill {margin-left:auto; font-size:.78rem; font-weight:600; color:#047857; background:#ecfdf5; border:1px solid #a7f3d0;
          padding:.25rem .7rem; border-radius:999px}
.pc-card {background:#fff; border:1px solid #e6e8ec; border-radius:16px; padding:18px 18px 14px;
          box-shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06)}
.pc-empty {text-align:center; padding:34px 18px} .pc-empty-icon {font-size:34px} .pc-empty-t {font-weight:700; font-size:1.05rem; margin:.3rem 0}
.pc-empty-sm {color:#64748b; text-align:center; padding:18px}
.pc-row {display:flex; align-items:center; gap:14px} .pc-grow {flex:1; min-width:0}
.pc-avatar {width:46px; height:46px; border-radius:14px; background:#ecfdf5; color:#047857; font-weight:700; font-size:1.2rem;
            display:flex; align-items:center; justify-content:center; flex:none}
.pc-avatar.pc-sm {width:36px; height:36px; border-radius:11px; font-size:1rem}
.pc-merchant {font-weight:700; font-size:1.08rem; color:#0f172a; overflow-wrap:anywhere}
.pc-muted {color:#64748b; font-size:.88rem}
.pc-amount {font-size:1.9rem; font-weight:800; color:#0f172a; text-align:right; line-height:1.05; letter-spacing:-.02em}
.pc-cur {display:block; font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8}
.pc-address {color:#334155; font-size:.9rem; margin:.7rem 0 .2rem; padding:.6rem .75rem; background:#f8fafc; border-radius:10px}
.pc-chips {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.6rem}
.pc-chip {font-size:.78rem; font-weight:600; padding:.22rem .65rem; border-radius:999px; background:#f1f5f9; color:#334155; border:1px solid #e2e8f0}
.pc-chip.pc-ok {background:#ecfdf5; color:#047857; border-color:#a7f3d0} .pc-chip.pc-warn {background:#fffbeb; color:#b45309; border-color:#fde68a}
.pc-note {color:#94a3b8; font-size:.75rem; margin-top:.55rem}
.pc-vendor {margin:.5rem 0 0; font-size:.85rem; font-weight:600; color:#334155; background:#f1f5f9; border:1px solid #e2e8f0;
            border-radius:12px; padding:.5rem .8rem} .pc-vendor.pc-ok {background:#ecfdf5; color:#047857; border-color:#a7f3d0}
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
.pc-move {display:flex; justify-content:space-between; padding:.5rem 0; border-top:1px solid #f1f5f9; color:#334155}
.pc-move span {font-weight:700; color:#0f172a}
.pc-foot {text-align:center; color:#94a3b8; font-size:.8rem; padding:12px 0 4px} .pc-foot a {color:#047857}
.r2e-h {font-weight:700; font-size:1rem; margin:.2rem 0} .r2e-chips {display:flex; gap:.4rem; flex-wrap:wrap; margin:.35rem 0 .5rem}
.r2e-chip {padding:.15rem .6rem; border-radius:999px; font-size:.8rem; background:#f1f5f9; border:1px solid #e2e8f0}
.r2e-scroll {overflow-x:auto} .r2e-table {width:100%; border-collapse:collapse; font-size:.88rem}
.r2e-table th, .r2e-table td {padding:.45rem .5rem; border-bottom:1px solid #eef2f6; text-align:left; vertical-align:top; overflow-wrap:anywhere}
.r2e-yes {color:#10b981; font-weight:700} .r2e-no {color:#ef4444; font-weight:700}
@media (max-width: 640px) { .pc-amount {font-size:1.5rem} .pc-title {font-size:1.2rem} .pc-pill {display:none} }
"""

with gr.Blocks(title="Receipt to Expense") as demo:
    gr.HTML('<div class="pc-appbar"><div class="pc-logo">&#129534;</div><div><div class="pc-title">Receipt to Expense</div>'
            '<div class="pc-sub">Snap a receipt &middot; check it &middot; split it. Fine-tuned Gemma 4 E2B, open model.</div></div>'
            '<div class="pc-pill">&#9679; Live</div></div>')
    split_state, upload_path = gr.State(None), gr.State(None)
    with gr.Tabs(selected="scan") as tabs:
        with gr.Tab("Scan", id="scan"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    img = gr.Image(type="filepath", sources=["upload", "webcam", "clipboard"], label="Receipt photo", height=380)
                    scan_btn = gr.Button("Scan receipt", variant="primary", size="lg")
                    gr.Examples(EXAMPLES, inputs=img, label="Or try a sample receipt (never used in training)")
                with gr.Column(scale=6, min_width=300):
                    card = gr.HTML(EMPTY_CARD)
                    vendor_html = gr.HTML()
                    with gr.Group():
                        with gr.Row():
                            company = gr.Textbox(label="Merchant", scale=3)
                            total = gr.Textbox(label="Total", scale=2)
                        with gr.Row():
                            date = gr.Textbox(label="Date", scale=2)
                            address = gr.Textbox(label="Address", scale=3)
                    with gr.Row():
                        add_btn = gr.Button("Add to expenses", variant="primary")
                        to_split_btn = gr.Button("Split this bill", variant="secondary")
            with gr.Accordion("Fine-tuned vs plain model, field by field", open=False):
                cmp_btn = gr.Button("Run the plain-model comparison", size="sm")
                compare = gr.HTML()
                with gr.Row():
                    ft_raw = gr.Textbox(label="Fine-tuned (raw)", lines=4)
                    plain_raw = gr.Textbox(label="Plain (raw)", lines=4)
        with gr.Tab("Split", id="split"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    with gr.Group():
                        people = gr.Textbox(label="Who's splitting?", value="Me, Riya, Arjun")
                        with gr.Row():
                            payer = gr.Textbox(label="Paid by", value="Me")
                            mode = gr.Radio(["Equally", "By items"], value="Equally", label="Split")
                    items = gr.Dataframe(headers=["item", "price", "shared by"], interactive=True, wrap=True,
                                         label="Items ('shared by': names or 'all')")
                    with gr.Row():
                        fill_btn = gr.Button("Auto-fill items", size="sm")
                        calc_btn = gr.Button("Calculate split", variant="primary")
                with gr.Column(scale=6, min_width=300):
                    split_out = gr.HTML('<div class="pc-card pc-empty-sm">Scan a receipt, then calculate the split here.</div>')
                    with gr.Row():
                        group_btn = gr.Button("Add to group", variant="primary")
                        reset_btn = gr.Button("Reset group", size="sm")
                    group_out = gr.HTML(_render_group())
        with gr.Tab("Expenses", id="expenses"):
            kpis = gr.HTML()
            table = gr.Dataframe(headers=["merchant", "date", "total", "address"], interactive=False, wrap=True, label="Expenses")
            with gr.Row():
                undo_btn = gr.Button("Undo last", size="sm")
                dl_btn = gr.DownloadButton("Download CSV", interactive=False, size="sm")
    gr.HTML(f'<div class="pc-foot">Open model: Gemma 4 E2B + one LoRA adapter file &middot; no cloud OCR, no subscription &middot; '
            f'<a href="{HF_URL}" target="_blank" rel="noopener">model card</a> &middot; photos are not stored or used for training</div>')

    scan_btn.click(lambda: "", outputs=vendor_html, api_visibility="private").then(
        scan, inputs=img, outputs=[card, company, date, total, address, compare, ft_raw, plain_raw, upload_path],
        api_name="scan", concurrency_limit=1).success(vendor_badge, inputs=[company, address], outputs=vendor_html,
                                                      api_visibility="private")
    cmp_btn.click(compare_models, inputs=upload_path, outputs=[compare, ft_raw, plain_raw], api_name="compare", concurrency_limit=1)
    add_btn.click(add_expense, inputs=[company, date, total, address], outputs=[kpis, table, dl_btn], api_name="add_expense")
    undo_btn.click(undo_expense, outputs=[kpis, table, dl_btn], api_visibility="private")
    to_split_btn.click(lambda: gr.Tabs(selected="split"), outputs=tabs, api_visibility="private")
    fill_btn.click(autofill_items, inputs=[img, upload_path], outputs=[items, mode], api_name="autofill_items", concurrency_limit=1)
    calc_btn.click(calc_split, inputs=[people, payer, mode, items, total, company], outputs=[split_out, split_state], api_name="calc_split")
    group_btn.click(add_to_group, inputs=split_state, outputs=group_out, api_name="add_to_group")
    reset_btn.click(reset_group, outputs=group_out, api_visibility="private")
    demo.load(_render_expenses, outputs=[kpis, table, dl_btn], api_visibility="private")
    demo.load(_render_group, outputs=group_out, api_visibility="private")
demo.queue(default_concurrency_limit=4, max_size=32)

def _warm():
    for f in (_cl, _fast_client):
        try:
            f()
        except Exception as e:
            print("warm-up:", f.__name__, type(e).__name__, flush=True)

if __name__ == "__main__":
    threading.Thread(target=_warm, daemon=True).start()     # connect to the model server before the first scan
    port = int(os.environ.get("R2E_PORT", "7871"))
    _, local, share = demo.launch(share=os.environ.get("R2E_SHARE", "1") == "1", server_name="127.0.0.1", server_port=port,
                                  theme=THEME, css=CSS, prevent_thread_lock=True, quiet=True)
    print("LOCAL:", local, "| PUBLIC:", share, flush=True)
    while True:
        time.sleep(3600)
