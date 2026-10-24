# %%
%%writefile /content/hackday/run_experiment.py
# CELL: Write run_experiment.py to the runtime (~1 s). Keep %%writefile as the first line of the Colab cell.
#!/usr/bin/env python3
# run_experiment.py: one background experiment for Theme 3B (SROIE receipts, Gemma 4 vision LoRA with Unsloth).
#
# Mirrors theme3_3B_receipts.ipynb as it was run on the Colab G4 runtime, including the runtime fixes:
#   * training images go in an "images" column and the collator is UnslothVisionDataCollator(resize="max"), so the
#     training pixels equal the inference pixels (GATE B2 pixel-parity check; aborts on mismatch);
#   * visual budget (--budget, default 560) applied to BOTH image_processor.max_soft_tokens (collator path) and
#     images_kwargs={"max_soft_tokens": budget} (inference); GATE B asserts 0.8*budget < image tokens <= budget;
#   * same INSTRUCTION / PROMPT_DETAILED, greedy decoding at batch size 1, MAX_NEW_TOKENS 256, load_in_4bit=False,
#     same seed-0 test subset (checked against <out>/test_keys.json written by the notebook);
#   * default collator labels (instruction + JSON answer trained, image and pad tokens masked), as seen at GATE C;
#   * a --smoke-steps smoke trainer, then the main trainer on the same model ("5 smoke + N", as in the notebook).
# Training data: the SROIE TRAIN split only (optionally a seeded fraction). Never the test split or local receipts.
#
# Outputs in <out>/exp_<tag>/: eval_<config>.csv (same columns as the notebook), summary.json (rewritten after every
# phase), lora_adapter/ (save_pretrained, never merged), outputs/checkpoint-*/, comparison.csv, DONE or FAILED.
#
# Examples:
#   python -u run_experiment.py --model unsloth/gemma-4-E4B-it --tag e4b_r32_s157_b560 --configs plain,prompted,finetuned
#   python -u run_experiment.py --tag e2b_r8_s157_b560 --rank 8 --alpha 8 --configs finetuned
#   python -u run_experiment.py --model unsloth/gemma-4-E4B-it --tag speed_e4b --speed-only 20   # on an idle GPU
import argparse, difflib, json, math, os, random, re, sys, time, traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from dateutil import parser as dparser
except ImportError:  # only for local --help dry runs; python-dateutil is installed on the Colab runtime
    dparser = None

# ---- section 1 settings, shared by every compared configuration ----
LOAD_IN_4BIT = False
MAX_SOFT_TOKENS = 560      # set from --budget in run()
SEED = 0                   # data seed (test subset, --train-frac subset); the training seed is --seed
MAX_NEW_TOKENS = 256
FIELDS = ["company", "date", "address", "total"]
INSTRUCTION = "Extract the company, date, address and total from this receipt. Reply with JSON only."
PROMPT_DETAILED = (
    "Extract four fields from this receipt. Reply with one JSON object only, with exactly these keys: "
    "company, date, address, total.\n"
    "Rules:\n"
    "- company: the business name exactly as printed, in capital letters.\n"
    "- date: the transaction date exactly as printed (keep its format, for example 25/12/2018 or 12-01-19).\n"
    "- address: the full business address exactly as printed, lines joined with a single space, in capital letters.\n"
    "- total: the final amount paid, digits with two decimals (for example 9.00), no currency symbol.\n"
    "No explanation, no markdown."
)
DATASET = "jsdnrs/ICDAR2019-SROIE"
CONFIGS = ("plain", "prompted", "finetuned")
GATE_B2_RECEIPTS = (0, 77, 282)   # small (463 px), medium (944 px) and huge (4961 px) training receipts

# Set in run(); module-level so the notebook helpers below work unchanged.
torch = FastVisionModel = model = processor = IMAGE_TOKEN_ID = binomtest = None
EXP_DIR, TAG = None, "?"
_budget_mode = None


def log(*parts):
    print(time.strftime("%H:%M:%S"), f"[{TAG}]", *parts, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One background Theme 3B experiment (mirrors theme3_3B_receipts.ipynb).")
    p.add_argument("--model", default="unsloth/gemma-4-E2B-it", help="base model")
    p.add_argument("--tag", required=True, help="experiment name; outputs go to <out>/exp_<tag>/")
    p.add_argument("--rank", type=int, default=32, help="LoRA r")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default: same as --rank)")
    p.add_argument("--steps", type=int, default=157,
                   help="main training steps; 0 = one epoch of the training (sub)set at batch 1 x grad-accum 4")
    p.add_argument("--smoke-steps", type=int, default=5, help="smoke trainer steps run first on the same model (notebook: 5)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=3407, help="LoRA random_state and SFT seed (notebook: 3407)")
    p.add_argument("--budget", type=int, default=560, help="visual token budget max_soft_tokens (70/140/280/560/1120)")
    p.add_argument("--train-frac", type=float, default=1.0, help="seeded (seed 0) fraction of the 626 training receipts")
    p.add_argument("--configs", default="plain,prompted,finetuned", help="comma list from plain, prompted, finetuned")
    p.add_argument("--n-test", type=int, default=100, help="must match <out>/test_keys.json when that file exists")
    p.add_argument("--save-steps", type=int, default=40, help="checkpoint every N main-run steps; 0 = no checkpoints")
    p.add_argument("--out", default="/content/hackday")
    p.add_argument("--adapter", default=None,
                   help="evaluate a saved adapter folder (or trainer checkpoint) instead of training; "
                        "needs --configs finetuned, or --speed-only")
    p.add_argument("--speed-only", type=int, default=0, metavar="N",
                   help="only time N test receipts (after one warm-up) and record memory; run it on an idle GPU")
    p.add_argument("--mem-frac", type=float, default=0.25, help="cap this process at this fraction of GPU memory; 0 = no cap")
    p.add_argument("--gpu-shared", choices=["yes", "no", "auto"], default="yes",
                   help='summary.json "gpu_shared"; auto = measured memory of other processes > 2 GB')
    p.add_argument("--force", action="store_true", help="re-run even if <out>/exp_<tag>/DONE exists")
    a = p.parse_args(argv)
    a.alpha = a.rank if a.alpha is None else a.alpha
    a.configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    bad = [c for c in a.configs if c not in CONFIGS]
    if bad:
        p.error(f"unknown configs {bad}; choose from {list(CONFIGS)}")
    if not a.configs and not a.speed_only:
        p.error("--configs is empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag):
        p.error("--tag may only contain letters, digits, '.', '_' and '-'")
    if not 0 < a.train_frac <= 1:
        p.error("--train-frac must be in (0, 1]")
    if not 1 <= a.n_test <= 361:
        p.error("--n-test must be in 1..361")
    if not 0 <= a.mem_frac <= 1:
        p.error("--mem-frac must be in [0, 1]")
    if min(a.budget, a.rank, a.alpha) <= 0 or min(a.steps, a.smoke_steps, a.save_steps, a.speed_only) < 0 or a.lr <= 0:
        p.error("--budget, --rank, --alpha and --lr must be > 0; --steps, --smoke-steps, --save-steps, --speed-only >= 0")
    if a.adapter and not a.speed_only and a.configs != ["finetuned"]:
        p.error("--adapter evaluates a saved adapter: use it with --configs finetuned (or with --speed-only)")
    return a


# ---- section 3 helpers (to_conversation as changed on the runtime: image in an "images" column) ----
def target_json(sample):
    return json.dumps({k: sample["entities"][k] for k in FIELDS}, ensure_ascii=False)

def to_conversation(sample):
    return {"messages": [
        {"role": "user", "content": [{"type": "image"},                                           # IMAGE FIRST
                                     {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_json(sample)}]},
    ], "images": [sample["image"].convert("RGB")]}


# ---- section 4 generation helpers (as in the notebook; MAX_SOFT_TOKENS comes from --budget) ----
def _encode(image, instruction):
    """Builds model inputs with the image first and tries to apply MAX_SOFT_TOKENS."""
    global _budget_mode
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    img = image.convert("RGB")
    tries = [("images_kwargs", {"images_kwargs": {"max_soft_tokens": MAX_SOFT_TOKENS}}),
             ("max_soft_tokens", {"max_soft_tokens": MAX_SOFT_TOKENS}),
             ("default", {})]
    if _budget_mode is not None:
        tries = [t for t in tries if t[0] == _budget_mode]
    last_err = None
    for name, kw in tries:
        try:
            inputs = processor(img, prompt, add_special_tokens=False, return_tensors="pt", **kw)
            _budget_mode = name
            return inputs
        except (TypeError, ValueError) as e:
            last_err = e
    raise last_err

def n_image_tokens(inputs):
    return None if IMAGE_TOKEN_ID is None else int((inputs["input_ids"] == IMAGE_TOKEN_ID).sum())

def generate_from_image(image, instruction=INSTRUCTION, max_new_tokens=MAX_NEW_TOKENS):
    with torch.inference_mode():                     # the notebook decorates with @torch.inference_mode()
        inputs = _encode(image, instruction).to("cuda")
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)   # greedy, batch 1
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---- section 5 scorer, verbatim from the notebook ----
def parse_json(text):
    text = re.sub(r"```(?:json)?", "", str(text))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None

def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().upper().strip(" .,;:")

def norm_total(s):
    nums = re.findall(r"\d+(?:\.\d+)?", str(s or "").replace(",", ""))
    return f"{float(nums[-1]):.2f}" if nums else ""

def norm_date(s):
    s = str(s or "").strip()
    if not s:
        return ""
    try:
        return dparser.parse(s, dayfirst=True, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return re.sub(r"\D", "", s)

NORMALIZERS = {"company": norm_text, "address": norm_text, "date": norm_date, "total": norm_total}
METRICS = ["valid_json"] + [f"{k}_{m}" for k in FIELDS for m in ("strict", "norm")] + ["all4_strict", "all4_norm"]

def score(prediction, gold):
    pred = parse_json(prediction)
    row = {"valid_json": int(pred is not None)}
    pred = pred or {}
    for k in FIELDS:
        p, g = str(pred.get(k, "") or "").strip(), str(gold[k] or "").strip()
        row[f"{k}_strict"] = int(p == g)
        row[f"{k}_norm"] = int(NORMALIZERS[k](g) != "" and NORMALIZERS[k](p) == NORMALIZERS[k](g))
        if k in ("company", "address"):
            row[f"{k}_sim"] = round(difflib.SequenceMatcher(None, norm_text(p), norm_text(g)).ratio(), 3)
    row["all4_strict"] = int(all(row[f"{k}_strict"] for k in FIELDS))
    row["all4_norm"] = int(all(row[f"{k}_norm"] for k in FIELDS))
    return row

def evaluate(samples, label, instruction=INSTRUCTION, plain=False):
    """The notebook's evaluate(): same loop and CSV columns; writes to the experiment folder and returns s/receipt."""
    import pandas as pd
    FastVisionModel.for_inference(model)
    rows, t0 = [], time.time()
    for i, s in enumerate(samples, 1):
        if plain and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                pred = generate_from_image(s["image"], instruction)
        else:
            pred = generate_from_image(s["image"], instruction)
        gold = s["entities"]
        rows.append({"key": s.get("key", i), "prediction": pred,
                     "expected": json.dumps({k: gold[k] for k in FIELDS}, ensure_ascii=False), **score(pred, gold)})
        if i % 10 == 0 or i == len(samples):
            log(f"{label}: {i}/{len(samples)} ({(time.time() - t0) / i:.1f} s/receipt)")
    s_per_receipt = (time.time() - t0) / len(samples)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, f"eval_{label}.csv"), index=False)
    log(f"{label} (%):\n" + df[METRICS].mean().mul(100).round(1).to_string())
    return df, s_per_receipt


# ---- section 12 McNemar, verbatim from the notebook ----
def mcnemar(base, new):
    better = int(((base == 0) & (new == 1)).sum()); worse = int(((base == 1) & (new == 0)).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    return better, worse, p


# ---- gates ----
def make_collator():
    from unsloth.trainer import UnslothVisionDataCollator
    return UnslothVisionDataCollator(model, processor, resize="max")   # default labels, as in the notebook (GATE C)

def check_pixel_parity(collator, train_data):
    """GATE B2: the training collator must feed exactly the inference pixels and image-token count."""
    rows = []
    for idx in GATE_B2_RECEIPTS:
        b = collator([to_conversation(train_data[idx])])
        i = _encode(train_data[idx]["image"], INSTRUCTION)
        k_train, k_infer = int((b["input_ids"] == IMAGE_TOKEN_ID).sum()), n_image_tokens(i)
        pt, pi = b["pixel_values"], i["pixel_values"]
        if pt.shape != pi.shape:
            raise RuntimeError(f"GATE B2 failed on receipt {idx}: pixel_values {tuple(pt.shape)} vs {tuple(pi.shape)}")
        px = (pt.float().cpu() - pi.to(pt.dtype).float().cpu()).abs().max().item()
        log(f"GATE B2 receipt {idx}: image tokens, training collator: {k_train} | inference: {k_infer} "
            f"| max pixel difference: {px:.4f}")
        rows.append({"receipt": idx, "train_tokens": k_train, "infer_tokens": k_infer, "max_pixel_diff": round(px, 5)})
        if k_train != k_infer or px > 1e-3:
            raise RuntimeError(f"GATE B2 failed on receipt {idx}: training and inference see different images")
    return rows

def check_labels(collator, conversation, answer):
    """GATE C: the trained tokens must contain the JSON answer (the notebook's default collator also trains the
    instruction and masks image tokens; both are recorded)."""
    batch = collator([conversation])
    labels = batch["labels"][0]
    kept = labels[labels != -100]
    n_img = int((batch["input_ids"] == IMAGE_TOKEN_ID).sum())
    n_img_trained = int((kept == IMAGE_TOKEN_ID).sum())
    log("GATE C: sequence length:", batch["input_ids"].shape[1], "| image tokens:", n_img,
        "| trained tokens:", kept.numel(), "| image tokens among trained:", n_img_trained)
    log("GATE C trained text:", processor.tokenizer.decode(kept[:300]))
    full = re.sub(r"\s+", "", processor.tokenizer.decode(kept))
    core = re.sub(r"\s+", "", answer)[2:-2]          # tolerant of token merges at the answer's edges
    if kept.numel() == 0 or core not in full:
        raise RuntimeError("GATE C failed: the trained tokens do not contain the JSON answer")
    instruction_trained = re.sub(r"\s+", "", INSTRUCTION) in full
    if n_img_trained or not instruction_trained:
        log("WARNING: labels differ from the notebook's GATE C (instruction trained, image tokens masked)")
    return {"sequence_length": int(batch["input_ids"].shape[1]), "image_tokens": n_img, "trained_tokens": int(kept.numel()),
            "image_tokens_trained": n_img_trained, "instruction_trained": instruction_trained, "answer_trained": True}


# ---- bookkeeping ----
def save_summary(summary):
    path = os.path.join(EXP_DIR, "summary.json")
    with open(path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    os.replace(path + ".tmp", path)

def gpu_note(summary, phase, args):
    free, total = torch.cuda.mem_get_info()
    used, mine = (total - free) / 1e9, torch.cuda.memory_reserved() / 1e9
    others = max(0.0, used - mine - 0.6)            # 0.6 GB ~ this process's CUDA context (approximation)
    summary.setdefault("gpu_memory_by_phase", {})[phase] = {
        "device_used_gb": round(used, 1), "this_process_reserved_gb": round(mine, 1), "others_gb_approx": round(others, 1)}
    if args.gpu_shared == "auto":
        summary["gpu_shared"] = bool(summary.get("gpu_shared")) or others > 2.0

def package_versions():
    from importlib.metadata import version
    out = {}
    for name in ("torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "datasets"):
        try:
            out[name] = version(name)
        except Exception:
            out[name] = None
    return out

def logged_losses(trainer):
    return [round(float(h["loss"]), 4) for h in trainer.state.log_history if "loss" in h]

def load_model(args, summary):
    kw = dict(load_in_4bit=LOAD_IN_4BIT, use_gradient_checkpointing="unsloth")
    if not args.adapter:
        summary["load_route"] = "base model"
        return FastVisionModel.from_pretrained(args.model, **kw)
    if any(os.path.exists(os.path.join(args.adapter, f)) for f in ("preprocessor_config.json", "processor_config.json")):
        summary["load_route"] = "FastVisionModel.from_pretrained(adapter folder)"   # Unsloth's documented reload
        return FastVisionModel.from_pretrained(args.adapter, **kw)
    from peft import PeftModel                       # trainer checkpoint: adapter only, processor from --model
    summary["load_route"] = "base model + PeftModel.from_pretrained(checkpoint)"
    base, proc = FastVisionModel.from_pretrained(args.model, **kw)
    return PeftModel.from_pretrained(base, args.adapter), proc

def speed_test(samples, n):
    """Clean speed numbers: run on an idle GPU. samples[0] was already generated once (warm-up)."""
    n = min(n, len(samples) - 1)
    torch.cuda.reset_peak_memory_stats()
    total, gen, toks = [], [], []
    for s in samples[1:n + 1]:
        torch.cuda.synchronize(); t0 = time.time()
        inputs = _encode(s["image"], INSTRUCTION).to("cuda")
        torch.cuda.synchronize(); t1 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=False)
        torch.cuda.synchronize(); t2 = time.time()
        processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        total.append(time.time() - t0); gen.append(t2 - t1); toks.append(int(out.shape[1] - inputs["input_ids"].shape[1]))
    res = {"receipts": n, "s_per_receipt_mean": round(sum(total) / n, 2), "s_per_receipt_median": round(sorted(total)[n // 2], 2),
           "generate_s_mean": round(sum(gen) / n, 2), "new_tokens_mean": round(sum(toks) / n, 1),
           "tokens_per_s": round(sum(toks) / sum(gen), 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    log("speed:", res)
    return res

def compare(args, results):
    """comparison.csv over this run's configs and the notebook's CSVs (main_* = E2B, r32, budget 560, 5 smoke + 157
    steps), plus paired exact McNemar tests of this run's fine-tuned model against each of them."""
    global binomtest
    import pandas as pd
    from scipy.stats import binomtest
    frames = dict(results)
    for label in CONFIGS:
        path = os.path.join(args.out, f"eval_{label}.csv")
        if os.path.exists(path):
            frames[f"main_{label}"] = pd.read_csv(path)
    if not frames:
        return []
    comp = pd.DataFrame({f"{name} (%)": df[METRICS].mean() * 100 for name, df in frames.items()}).round(1)
    comp.to_csv(os.path.join(EXP_DIR, "comparison.csv"))
    log("comparison (main_* = notebook run: E2B, r32, budget 560, 5 smoke + 157 steps):\n" + comp.to_string())
    tests, new = [], results.get("finetuned")
    if new is None:
        return tests
    for name, base in frames.items():
        if name == "finetuned":
            continue
        m = new[["key", "all4_norm", "all4_strict"]].merge(base[["key", "all4_norm", "all4_strict"]], on="key",
                                                            suffixes=("_new", "_base"))
        if len(m) != len(new) or len(base) != len(new):
            tests.append({"finetuned_vs": name, "skipped": f"{len(m)} shared keys of {len(new)}"})
            continue
        for metric in ("all4_norm", "all4_strict"):
            b, w, p = mcnemar(m[f"{metric}_base"].values, m[f"{metric}_new"].values)
            tests.append({"finetuned_vs": name, "metric": metric, "improved": b, "worse": w, "p": float(p)})
            log(f"fine-tuned ({TAG}) vs {name} on {metric}: {b} receipts improved, {w} got worse, exact McNemar p = {p:.2g}")
    return tests


def train_model(args, summary, train_rows, steps):
    """Sections 8-11: LoRA, GATE C, smoke trainer (GATE D), main trainer, save adapters straight away."""
    global model
    from trl import SFTTrainer, SFTConfig
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none", random_state=args.seed,
    )
    t = time.time()
    train_conversations = [to_conversation(s) for s in train_rows]
    log(f"built {len(train_conversations)} training conversations in {time.time() - t:.0f} s")
    collator = make_collator()
    summary["gate_c"] = check_labels(collator, train_conversations[0], target_json(train_rows[0]))
    save_summary(summary)

    def make_trainer(max_steps, out_dir, save=False):
        return SFTTrainer(
            model=model, train_dataset=train_conversations, processing_class=processor.tokenizer,
            data_collator=collator,
            args=SFTConfig(
                per_device_train_batch_size=1, gradient_accumulation_steps=4, max_grad_norm=0.3,
                warmup_ratio=0.03, max_steps=max_steps, learning_rate=args.lr, logging_steps=1 if max_steps <= 10 else 5,
                optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="cosine", seed=args.seed,
                output_dir=out_dir, report_to="none",
                save_strategy="steps" if save else "no", save_steps=max(1, args.save_steps),
                remove_unused_columns=False, dataset_text_field="",                 # required for vision fine-tuning
                dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048,
            ),
        )

    torch.cuda.reset_peak_memory_stats()
    if args.smoke_steps > 0:                          # GATE D, on the same model as the main run (notebook section 9)
        FastVisionModel.for_training(model)
        trainer = make_trainer(args.smoke_steps, os.path.join(EXP_DIR, "smoke"))
        res = trainer.train()
        losses = logged_losses(trainer)
        sps = res.metrics["train_runtime"] / args.smoke_steps
        summary["smoke"] = {"steps": args.smoke_steps, "s_per_step": round(sps, 2), "losses": losses}
        log(f"GATE D smoke: {sps:.1f} s/step (includes warm-up) -> {steps} steps ~ {sps * steps / 60:.0f} min | losses {losses}")
        if not losses or not all(math.isfinite(x) for x in losses):
            raise RuntimeError(f"GATE D failed: smoke losses {losses}")
        del trainer
        save_summary(summary)
    FastVisionModel.for_training(model)
    trainer = make_trainer(steps, os.path.join(EXP_DIR, "outputs"), save=args.save_steps > 0)
    res = trainer.train()
    adapter_dir = os.path.join(EXP_DIR, "lora_adapter")
    model.save_pretrained(adapter_dir)                # adapters only, never merged weights (merged output was garbled)
    processor.save_pretrained(adapter_dir)
    losses = logged_losses(trainer)
    with open(os.path.join(EXP_DIR, "train_log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1, default=str)
    runtime = res.metrics["train_runtime"]
    summary.update(train_runtime_s=round(runtime, 1), train_runtime_min=round(runtime / 60, 2),
                   s_per_step_full=round(runtime / steps, 2), adapter_dir=adapter_dir,
                   losses={"first": losses[0] if losses else None, "last": losses[-1] if losses else None,
                           "n_logged": len(losses)})
    summary["peak_gpu_gb"]["train"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    log(f"trained {steps} steps in {runtime / 60:.1f} min ({runtime / steps:.2f} s/step) | peak "
        f"{summary['peak_gpu_gb']['train']} GB | adapters saved to {adapter_dir}")
    gpu_note(summary, "train", args)
    save_summary(summary)
    if not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError(f"non-finite training losses: {losses[:5]} ... {losses[-5:]}")
    del trainer


def run(args, summary):
    global torch, FastVisionModel, model, processor, IMAGE_TOKEN_ID, MAX_SOFT_TOKENS
    import unsloth  # noqa: F401  (before transformers / trl / peft, so Unsloth can patch them)
    from unsloth import FastVisionModel as _FastVisionModel
    import torch as _torch
    from datasets import load_dataset
    FastVisionModel, torch = _FastVisionModel, _torch
    MAX_SOFT_TOKENS = args.budget

    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    summary.update(
        model=args.model, adapter=args.adapter, mode="speed-only" if args.speed_only else "experiment",
        load_in_4bit=LOAD_IN_4BIT, visual_budget=args.budget, budget_mode=None, instruction=INSTRUCTION,
        decoding="greedy (do_sample=False), batch size 1", max_new_tokens=MAX_NEW_TOKENS,
        lora_r=args.rank, lora_alpha=args.alpha, lr=args.lr, seed=args.seed, smoke_steps=args.smoke_steps,
        batch="1 x grad-accum 4", data_seed=SEED,
        collator='UnslothVisionDataCollator(model, processor, resize="max"), images column, default labels',
        configs=args.configs, gpu=torch.cuda.get_device_name(0), gpu_shared=args.gpu_shared == "yes",
        mem_frac_cap=args.mem_frac, versions=package_versions(), s_per_receipt={}, metrics={}, peak_gpu_gb={})
    gpu_note(summary, "start", args)
    save_summary(summary)

    # ---- section 2: load ----
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model, processor = load_model(args, summary)
    try:
        from unsloth.chat_templates import get_chat_template
        processor = get_chat_template(processor, "gemma-4")      # as in the official Unsloth Gemma 4 vision notebook
    except Exception as e:
        log("chat-template step skipped:", repr(e)[:200])
    ip = getattr(processor, "image_processor", None)
    if ip is None or not hasattr(ip, "max_soft_tokens"):
        raise RuntimeError(f"{type(ip).__name__} has no max_soft_tokens: the visual budget cannot be applied")
    ip.max_soft_tokens = MAX_SOFT_TOKENS              # collator (training) path; inference also passes images_kwargs
    IMAGE_TOKEN_ID = getattr(processor, "image_token_id", None)
    if IMAGE_TOKEN_ID is None:
        raise RuntimeError("processor.image_token_id is missing")
    summary.update(load_s=round(time.time() - t, 1), weights_gb=round(torch.cuda.memory_allocated() / 1e9, 2))
    summary["peak_gpu_gb"]["load"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    log(f"loaded {args.adapter or args.model} ({summary['load_route']}) in {summary['load_s']} s | "
        f"{type(ip).__name__}.max_soft_tokens = {ip.max_soft_tokens} | image_token_id {IMAGE_TOKEN_ID} | "
        f"{summary['weights_gb']} GB allocated")
    gpu_note(summary, "load", args)
    save_summary(summary)

    # ---- section 3: data and the fixed test subset ----
    train_data = load_dataset(DATASET, split="train")
    test_data = load_dataset(DATASET, split="test")
    log(len(train_data), "train /", len(test_data), "test receipts")
    keys_path = os.path.join(args.out, "test_keys.json")
    saved_keys = None
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            saved_keys = json.load(f)
    n_test = len(saved_keys) if (args.speed_only and saved_keys) else args.n_test
    test_idx = sorted(random.Random(SEED).sample(range(len(test_data)), n_test))
    test_subset = [test_data[i] for i in test_idx]
    test_keys = [s["key"] for s in test_subset]
    if saved_keys is not None and test_keys != saved_keys:
        raise RuntimeError(f"test subset ({len(test_keys)} keys) differs from {keys_path} ({len(saved_keys)} keys): "
                           "results would not be paired with the notebook")
    log(f"test subset: {n_test} receipts, " + (f"identical to {keys_path}" if saved_keys else f"WARNING: no {keys_path}"))
    with open(os.path.join(EXP_DIR, "test_keys.json"), "w") as f:
        json.dump(test_keys, f)
    assert not set(train_data["key"]) & set(test_keys), "test receipt found in training data"
    train_rows = train_data
    if args.train_frac < 1:
        n_train = max(4, round(args.train_frac * len(train_data)))
        train_rows = train_data.select(sorted(random.Random(SEED).sample(range(len(train_data)), n_train)))
    steps = args.steps if args.steps > 0 else math.ceil(len(train_rows) / 4)
    train_planned = "finetuned" in args.configs and not args.adapter and not args.speed_only
    summary["test_receipts"] = n_test
    if train_planned:
        summary.update(train_receipts=len(train_rows), steps=f"{args.smoke_steps} smoke + {steps}", main_steps=steps,
                       main_epochs=round(steps * 4 / len(train_rows), 2))

    # ---- section 4: GATE B (budget applied, output looks like JSON; also the compile warm-up) ----
    FastVisionModel.for_inference(model)
    probe = _encode(test_subset[0]["image"], INSTRUCTION)
    k = n_image_tokens(probe)
    log("GATE B: budget mode:", _budget_mode, "| image tokens:", k, "| prompt length:", probe["input_ids"].shape[1])
    if _budget_mode != "images_kwargs" or k is None or not 0.8 * MAX_SOFT_TOKENS < k <= MAX_SOFT_TOKENS:
        raise RuntimeError(f"GATE B failed: budget mode {_budget_mode}, {k} image tokens "
                           f"(need images_kwargs and {0.8 * MAX_SOFT_TOKENS:.0f} < k <= {MAX_SOFT_TOKENS})")
    t = time.time()
    out = generate_from_image(test_subset[0]["image"])
    dt = time.time() - t
    log("GATE B output:", out.replace("\n", " ")[:300], f"| {dt:.1f} s (first call, includes warm-up)")
    log("GATE B expected:", target_json(test_subset[0]))
    summary["budget_mode"] = _budget_mode
    summary["gate_b"] = {"image_tokens": k, "prompt_length": int(probe["input_ids"].shape[1]), "first_call_s": round(dt, 1)}
    save_summary(summary)

    if args.speed_only:
        summary["speed"] = speed_test(test_subset, args.speed_only)
        summary["peak_gpu_gb"]["speed"] = summary["speed"]["peak_gpu_gb"]
        gpu_note(summary, "speed", args)
        return

    # ---- section 4b: GATE B2 before any baseline, if this run trains ----
    if train_planned:
        summary["gate_b2"] = check_pixel_parity(make_collator(), train_data)
        save_summary(summary)

    results = {}
    def run_eval(label, **kw):
        torch.cuda.reset_peak_memory_stats()
        df, s_per_receipt = evaluate(test_subset, label, **kw)
        results[label] = df
        summary["s_per_receipt"][label] = round(s_per_receipt, 2)
        summary["peak_gpu_gb"][label] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        metrics = {m: round(float(v), 1) for m, v in (df[METRICS].mean() * 100).items()}
        metrics.update({f"{k}_sim_mean": round(float(df[f"{k}_sim"].mean()), 3) for k in ("company", "address")})
        summary["metrics"][label] = metrics
        gpu_note(summary, label, args)
        save_summary(summary)

    # ---- sections 6-7: baselines on the base model, before LoRA ----
    if "plain" in args.configs:
        run_eval("plain")
    if "prompted" in args.configs:
        run_eval("prompted", instruction=PROMPT_DETAILED)
    # ---- sections 8-11: LoRA, smoke + main training, adapters saved ----
    if train_planned:
        train_model(args, summary, train_rows, steps)
    # ---- section 12: fine-tuned evaluation, comparison, McNemar ----
    if "finetuned" in args.configs:
        run_eval("finetuned")
    try:
        summary["paired_tests"] = compare(args, results)
    except Exception as e:                            # never fail a finished run over the statistics
        summary["paired_tests_error"] = repr(e)[:500]
        log("paired tests skipped:", repr(e)[:300])


def main(argv=None):
    global TAG, EXP_DIR
    args = parse_args(argv)
    TAG, EXP_DIR = args.tag, os.path.join(args.out, f"exp_{args.tag}")
    os.makedirs(EXP_DIR, exist_ok=True)
    done, failed = os.path.join(EXP_DIR, "DONE"), os.path.join(EXP_DIR, "FAILED")
    if os.path.exists(done) and not args.force:
        log(f"already finished ({done} exists): skipping; pass --force to re-run")
        return 0
    for marker in (done, failed):
        if os.path.exists(marker):
            os.remove(marker)
    summary = {"tag": args.tag, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
               "args": vars(args)}
    try:
        run(args, summary)
        summary.update(status="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_summary(summary)
        with open(done, "w") as f:
            json.dump({k: summary.get(k) for k in ("metrics", "s_per_receipt", "s_per_step_full", "peak_gpu_gb", "speed")},
                      f, indent=1, default=str)
        log("DONE:", EXP_DIR)
        return 0
    except BaseException:
        tb = traceback.format_exc()
        summary.update(status="failed", finished=time.strftime("%Y-%m-%d %H:%M:%S"), error=tb[-4000:])
        try:
            save_summary(summary)
        except Exception:
            pass
        with open(failed, "w") as f:
            f.write(tb)
        log("FAILED:\n" + tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# %%
# CELL: Launch 3 parallel background queues: Q1 E4B size, Q2 LoRA rank/steps, Q3 visual budget (~2 s; queues finish in ~60-80 min)
import os, shlex, subprocess, sys
BG_OUT = "/content/hackday"
BG_SCRIPT = f"{BG_OUT}/run_experiment.py"
assert os.path.exists(BG_SCRIPT), "run the %%writefile cell first"
assert os.path.exists(f"{BG_OUT}/test_keys.json"), "run notebook section 3 first (fixed 100-receipt test subset)"

def job(size, r, steps, budget, configs, seed=3407):
    tag = f"{size.lower()}_r{r}_s{steps}_b{budget}" + ("" if seed == 3407 else f"_seed{seed}")
    return (f"{sys.executable} -u {BG_SCRIPT} --model unsloth/gemma-4-{size}-it --tag {tag} --rank {r} --alpha {r} "
            f"--steps {steps} --budget {budget} --seed {seed} --configs {configs} --out {BG_OUT}")

def launch(name, mem_frac, jobs):
    """One queue = one detached nohup bash -c "job1 ; job2" with its own log, working dir and Unsloth compile cache."""
    if subprocess.run(["pgrep", "-f", f"BG_QUEUE={name};"], capture_output=True).returncode == 0:
        print(f"{name}: already running, not launched again")
        return
    work = f"{BG_OUT}/bg_work/{name}"
    os.makedirs(work, exist_ok=True)
    queue = (f"export BG_QUEUE={name}; " + " ; ".join(f"{j} --mem-frac {mem_frac}" for j in jobs)
             + f" ; echo QUEUE {name} FINISHED $(date)")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
               UNSLOTH_COMPILE_LOCATION=f"{work}/unsloth_compiled_cache")
    env.pop("MPLBACKEND", None)                  # the kernel's inline matplotlib backend does not exist outside it
    log = f"{BG_OUT}/bg_{name}.log"
    subprocess.Popen(f"cd {shlex.quote(work)} && nohup bash -c {shlex.quote(queue)} > {shlex.quote(log)} 2>&1 &",
                     shell=True, env=env, start_new_session=True)
    print(f"{name}: launched {len(jobs)} job(s), capped at {mem_frac:.0%} of GPU memory, log {log}")

# GPU caps (share of ~95 GB): 28.5 + 17.1 + 20.9 = 66.5 GB worst case. Expected peaks: E4B ~20, E2B 13 (measured),
# E2B at budget 1120 ~16 GB, plus the main kernel's 13-20 GB. A job that hits its cap fails alone; its queue moves on.
launch("q1_size", 0.30, [job("E4B", 32, 157, 560, "plain,prompted,finetuned")])
launch("q2_train", 0.18, [job("E2B", 8, 157, 560, "finetuned"), job("E2B", 32, 314, 560, "finetuned")])
launch("q3_budget", 0.22, [job("E2B", 32, 157, 1120, "plain,finetuned"), job("E2B", 32, 157, 280, "plain,finetuned")])
# Q4 (optional, seed variance; adds ~13-17 GB while it runs). Uncomment to launch:
# launch("q4_seeds", 0.18, [job("E2B", 32, 157, 560, "finetuned", seed=1), job("E2B", 32, 157, 560, "finetuned", seed=2)])

# %%
# CELL: Background status: each queue's log tail, DONE/FAILED per tag, key numbers, GPU and RAM use (~3 s)
import glob, json, os, subprocess
BG_OUT = "/content/hackday"
def sh(*cmd):
    return subprocess.run(list(cmd), capture_output=True, text=True).stdout
for log in sorted(glob.glob(f"{BG_OUT}/bg_q*.log")):
    lines = sh("tail", "-n", "6", log).replace("\r", "\n").strip().splitlines()
    print(f"===== {os.path.basename(log)}\n" + "\n".join(l[:220] for l in lines[-6:]))
jobs = [l for l in sh("pgrep", "-af", "run_experiment.py").splitlines() if "BG_QUEUE=" not in l]
for d in sorted(glob.glob(f"{BG_OUT}/exp_*")):
    tag = os.path.basename(d)[4:]
    state = ("DONE" if os.path.exists(f"{d}/DONE") else "FAILED" if os.path.exists(f"{d}/FAILED")
             else "running" if any(f"--tag {tag} " in j for j in jobs) else "STOPPED without a marker (killed? out of RAM?)")
    print(f"\n== {tag}: {state}")
    if state == "FAILED":
        print(open(f"{d}/FAILED").read()[-1500:])
    try:
        s = json.load(open(f"{d}/summary.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    phases = s.get("gpu_memory_by_phase") or {}
    last = list(phases.items())[-1] if phases else ("-", {})
    print(f"   s/receipt {s.get('s_per_receipt')} | s/step {s.get('s_per_step_full')} | train {s.get('train_runtime_min')} min"
          f" | peak GB {s.get('peak_gpu_gb')} | reserved now ({last[0]}) {last[1].get('this_process_reserved_gb')} GB")
    for cfg, m in (s.get("metrics") or {}).items():
        print(f"   {cfg:9s} all4_norm {m.get('all4_norm')} | all4_strict {m.get('all4_strict')} | valid_json {m.get('valid_json')}")
    for t in s.get("paired_tests") or []:
        if t.get("metric") == "all4_norm":
            print(f"   fine-tuned vs {t['finetuned_vs']}: +{t['improved']} / -{t['worse']} receipts, McNemar p = {t['p']:.2g}")
print("\n===== GPU memory per process (can be empty inside the Colab container; see 'reserved now' above):")
print(sh("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"))
print(sh("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"))
print(sh("free", "-g"))
