"""Headless smoke test for app_cell.py with stubbed model functions (no GPU, no share link).
Runs cell 15a in a namespace that mimics the Colab kernel after sections 0-12, using the notebook's real section-5 scorer.
Run: .venv/Scripts/python.exe -I smoke_test.py
"""
import contextlib, json, os, re, sys, time, types, urllib.request
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import gradio as gr
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_smoke_out")
NOTEBOOK = os.path.join(HERE, "..", "..", "theme3_3B_receipts.ipynb")
INSTRUCTION = "Extract the company, date, address and total from this receipt. Reply with JSON only."
FIELDS = ["company", "date", "address", "total"]
FT = {"company": "OJC MARKETING SDN BHD", "date": "15/01/2019",
      "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR", "total": "193.00"}
PLAIN = ('```json\n{"company": "OJC Marketing Sdn Bhd", "date": "2019-01-15", '
         '"address": "No 2 & 4, Jalan Bayu 4", "total": "RM 193.00"}\n```')
checks = []

def check(name, ok):
    checks.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name)

class StubModel:
    def __init__(self):
        self.adapter_on, self.calls = True, []
    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True

def make_generate(m, mode=None):
    def generate_from_image(image, instruction=INSTRUCTION, max_new_tokens=256):
        m.calls.append(("ft" if m.adapter_on else "plain", instruction))
        time.sleep(0.02)
        if m.adapter_on:
            if mode == "fail_ft":
                raise RuntimeError("CUDA out of memory (stub)")
            return "sorry, no JSON here" if mode == "bad_json" else json.dumps(FT)
        if mode == "fail_plain":
            raise RuntimeError("stub plain failure")
        return PLAIN
    return generate_from_image

def receipt(text, size=(420, 760)):
    im = Image.new("RGB", size, "white"); d = ImageDraw.Draw(im)
    for i, line in enumerate(text.split("\n")):
        d.text((20, 20 + 22 * i), line, fill="black")
    return im

def scorer_source():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return next(s for s in src if s.startswith("import re, difflib"))      # section 5, verbatim

def app_cells():
    text = open(os.path.join(HERE, "app_cell.py"), encoding="utf-8").read()
    cells = [c for c in re.split(r"^# %%\s*$", text, flags=re.M) if c.strip()]
    assert len(cells) == 2, len(cells)
    return cells

def namespace(extras=True, mode=None, hf_repo="team/gemma4-e2b-sroie-lora"):
    m = StubModel()
    ns = {"__name__": "__colab__", "os": os, "re": re, "json": json, "time": time, "pd": pd,
          "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
          "FastVisionModel": types.SimpleNamespace(for_inference=lambda model: None),
          "model": m, "generate_from_image": make_generate(m, mode), "INSTRUCTION": INSTRUCTION, "OUT": OUT, "FIELDS": FIELDS,
          "LORA_R": 32, "MAX_SOFT_TOKENS": 560, "HF_REPO": hf_repo,
          "test_subset": [{"image": receipt("OJC MARKETING SDN BHD\nTOTAL 193.00"), "entities": dict(FT)},
                          {"image": receipt("SHOP B\nTOTAL 9.00"), "entities": {**FT, "company": "SHOP B", "total": "9.00"}},
                          {"image": receipt("SHOP C\nTOTAL 60.30"), "entities": {**FT, "company": "SHOP C", "total": "60.30"}}]}
    if extras:
        ns["extract_line_items"] = lambda image, use_adapter=False: (
            [{"name": "ITEM A", "quantity": "1", "price": "100.00"}, {"name": "ITEM B", "quantity": 2, "price": 93}], "[...]", 0.4)
        ns["vendor_lookup"] = lambda company, address: {
            "company": FT["company"], "address": FT["address"], "similarity": 0.93, "matched": True}
    os.makedirs(OUT, exist_ok=True)
    exec(scorer_source(), ns)
    exec(app_cells()[0], ns)
    return ns, m

# ---------- A. full build, handlers called directly ----------
for f in ("expenses.csv",):
    if os.path.exists(os.path.join(OUT, f)):
        os.remove(os.path.join(OUT, f))
ns, m = namespace()
check("A1 demo is gr.Blocks", isinstance(ns["demo"], gr.Blocks))
ex_path = os.path.join(OUT, "examples", "receipt_0.jpg")
check("A2 3 example files written", all(os.path.exists(os.path.join(OUT, "examples", f"receipt_{i}.jpg")) for i in range(3)))
out = ns["extract"](Image.open(ex_path))
check("A3 extract returns 8 outputs", len(out) == 8)
check("A4 fields from fine-tuned JSON", list(out[:4]) == [FT[k] for k in FIELDS])
check("A5 both models called with the SAME instruction", m.calls == [("ft", INSTRUCTION), ("plain", INSTRUCTION)])
check("A6 adapter re-enabled after plain call", m.adapter_on)
check("A7 example image matched its SROIE label", "SROIE label" in out[7])
check("A8 per-field marks present", "&#10003; match" in out[7] and "&#10007; differs" in out[7])
check("A9 latency chips", "Fine-tuned &middot;" in out[7] and "Plain &middot;" in out[7])
check("A10 ok callout when all fields parse", "r2e-callout ok" in out[6])
out2 = ns["extract"](receipt("SOME OTHER SHOP"))
check("A11 non-example image has no label column", "SROIE label" not in out2[7])
v_html, v_state, v_btn = ns["lookup_vendor"](FT["company"], FT["address"])
check("A12 vendor badge visible + matched", v_html.visible and "Known vendor" in v_html.value and v_state["matched"])
check("A13 apply button hidden when fields already equal canonical", v_btn.visible is False)
v_html, v_state, v_btn = ns["lookup_vendor"]("OJC MARKETING", "JALAN BAYU")
check("A14 apply button shown when fields differ", v_btn.visible is True)
check("A15 apply_vendor fills canonical values", ns["apply_vendor"](v_state, "x", "y") == (FT["company"], FT["address"]))
df, kpis, path = ns["add_expense"](*out[:4])
df, kpis, path = ns["add_expense"]("SHOP B", "01/02/2019", "", "9.00")
check("A16 two expenses in table", len(df) == 2)
check("A17 KPI total spend 202.00", "202.00" in kpis and ">2<" in kpis)
check("A18 CSV written with 2 rows", path and len(pd.read_csv(path)) == 2)
df, kpis, path = ns["undo_last"]()
check("A19 undo removes last row", len(df) == 1 and len(pd.read_csv(path)) == 1)
try:
    ns["add_expense"]("", "", "", ""); check("A20 empty add rejected", False)
except gr.Error:
    check("A20 empty add rejected", True)
li_df, li_sum, li_raw = ns["run_line_items"](Image.open(ex_path), "193.00")
check("A21 line items table + sum check", len(li_df) == 2 and "Line items (2)" in li_sum and "r2e-yes" in li_sum)
try:
    ns["extract"](None); check("A22 extract(None) -> gr.Error", False)
except gr.Error:
    check("A22 extract(None) -> gr.Error", True)
check("A23 HF model card link built", "huggingface.co/team/gemma4-e2b-sroie-lora" in ns["_hf_link"])
check("A24 reset_outputs returns 11 values", len(ns["reset_outputs"](None)) == 11)

# ---------- A'. failure modes ----------
nsf, _ = namespace(mode="fail_ft")
try:
    nsf["extract"](receipt("X")); check("A25 fine-tuned failure -> gr.Error", False)
except gr.Error:
    check("A25 fine-tuned failure -> gr.Error", True)
nsp, _ = namespace(mode="fail_plain")
o = nsp["extract"](receipt("X"))
check("A26 plain failure -> still returns, plain marked failed", o[0] == FT["company"] and "plain model failed" in o[5] and "failed" in o[7])
nsb, _ = namespace(mode="bad_json")
o = nsb["extract"](receipt("X"))
check("A27 bad JSON -> blank fields + warn callout", o[:4] == ("", "", "", "") and "r2e-callout warn" in o[6])

# ---------- B. launch locally, fetch the page, call over HTTP ----------
demo = ns["demo"]
app, local_url, share_url = demo.launch(share=False, prevent_thread_lock=True, server_name="127.0.0.1", quiet=True,
                                        theme=ns["APP_THEME"], css=ns["APP_CSS"], allowed_paths=[OUT], show_error=True)
html_page = urllib.request.urlopen(local_url, timeout=30).read().decode("utf-8", "replace")
check("B1 page served (HTTP 200)", len(html_page) > 500)
cfg = json.loads(urllib.request.urlopen(local_url.rstrip("/") + "/config", timeout=30).read())
check("B2 config title + custom CSS present", cfg.get("title") == "Receipt to Expense" and "r2e-kpis" in (cfg.get("css") or ""))
check("B3 share disabled in test", share_url is None)
from gradio_client import Client, handle_file
client = Client(local_url, verbose=False)
res = client.predict(handle_file(ex_path), api_name="/extract")
check("B4 /extract over HTTP (upload path)", res[0] == FT["company"] and "SROIE label" in res[7])
res = client.predict(FT["company"], "15/01/2019", FT["address"], "193.00", api_name="/add_expense")
check("B5 /add_expense over HTTP", "193.00" in res[1])
res = client.predict(FT["company"], "JALAN BAYU", api_name="/lookup_vendor")
check("B6 /lookup_vendor over HTTP", "Known vendor" in json.dumps(res))
res = client.predict(handle_file(ex_path), "193.00", api_name="/line_items")
check("B7 /line_items over HTTP", "Line items (2)" in res[1])
demo.close()
# what cell 15b does on a rerun: close, then launch again
try:
    demo.close()
except Exception:
    pass
app, local_url2, _ = demo.launch(share=False, prevent_thread_lock=True, server_name="127.0.0.1", quiet=True,
                                 theme=ns["APP_THEME"], css=ns["APP_CSS"], allowed_paths=[OUT])
check("B8 relaunch after close works", urllib.request.urlopen(local_url2, timeout=30).status == 200)
demo.close()

# ---------- C. build without the optional extras ----------
ns2, m2 = namespace(extras=False, hf_repo="")
check("C1 builds without extras", isinstance(ns2["demo"], gr.Blocks) and ns2["HAS_LINE_ITEMS"] is False)
v = ns2["lookup_vendor"](FT["company"], FT["address"])
check("C2 vendor badge hidden without vendor_lookup", v[0].visible is False and v[1] is None)
try:
    ns2["run_line_items"](receipt("X"), "1.00"); check("C3 line items -> gr.Error without function", False)
except gr.Error:
    check("C3 line items -> gr.Error without function", True)
check("C4 no HF link when HF_REPO empty", ns2["_hf_link"] == "")
check("C5 extract still works", ns2["extract"](receipt("X"))[0] == FT["company"])

failed = [n for n, ok in checks if not ok]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
