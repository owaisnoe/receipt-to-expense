# === PASTE THIS WHOLE CELL INTO THE ORIGINAL COLAB NOTEBOOK (new cell at the bottom) AND RUN IT (about 30-60 s) ===
# 1) Fix: a later cell reused the name _rows, which broke "Fill items" and "By items" in the app. Restore the helper.
def _rows(df):
    if df is None:
        return []
    if isinstance(df, dict):
        df = pd.DataFrame(df.get("data") or [], columns=(df.get("headers") or None))
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return [[_s(x) for x in r] for r in df.itertuples(index=False)]

# 2) Use the best version: the budget-1120 adapter (92% of fields, 76% of receipts all-correct). Swapped in place:
#    the existing public link keeps working and now serves this model too.
import re, torch
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
_LRE = re.compile(r"\.(lora_[A-Za-z_]+)\.default(\.|$)")
_lp = {n: p for n, p in model.named_parameters() if _LRE.search(n)}
_exp = {_LRE.sub(r".\1\2", n): n for n in _lp}
_sd = load_file(f"{OUT}/exp_e2b_r32_s157_b1120/lora_adapter/adapter_model.safetensors", device="cpu")
assert not (set(_exp) - set(_sd)), "LoRA keys differ: nothing was changed"
with GEN_LOCK:
    _res = set_peft_model_state_dict(model, {k: _sd[k] for k in _exp}, adapter_name="default")
    MAX_SOFT_TOKENS = 1120
    processor.image_processor.max_soft_tokens = 1120
    FastVisionModel.for_inference(model)
    _t = time.time(); _o = generate_from_image(test_subset[0]["image"]); _dt = time.time() - _t
print("best adapter loaded:", not _res.unexpected_keys, "| budget", MAX_SOFT_TOKENS, f"| test receipt in {_dt:.1f} s:", _o.replace("\n", " ")[:150])

# 3) Fast endpoint for the polished laptop UI: fine-tuned model only (no plain pass) + line items at budget 560.
def fast_extract(image):
    raw, t = _generate(image, adapter=True)
    f = parse_json(raw) or {}
    return _s(f.get("company")), _s(f.get("date")), _s(f.get("address")), _s(f.get("total")), raw, round(t, 2)

def fast_items(image):
    global MAX_SOFT_TOKENS
    with GEN_LOCK:
        _b, MAX_SOFT_TOKENS = MAX_SOFT_TOKENS, 560
        try:
            items, raw, secs = extract_line_items(image, use_adapter=False)
        finally:
            MAX_SOFT_TOKENS = _b
    return json.dumps(items, ensure_ascii=False), round(secs, 2)

with gr.Blocks() as fast_api:
    _fi = gr.Image(type="pil")
    _fo = [gr.Textbox() for _ in range(5)] + [gr.Number()]
    gr.Button("extract").click(fast_extract, _fi, _fo, api_name="fast_extract")
    _io = [gr.Textbox(), gr.Number()]
    gr.Button("items").click(fast_items, _fi, _io, api_name="fast_items")
fast_api.queue(default_concurrency_limit=1)
FAST_URL = fast_api.launch(share=True, prevent_thread_lock=True, server_port=7862, quiet=True)[2]
print("\nFAST_URL  ->  send this link to Claude:", FAST_URL)
