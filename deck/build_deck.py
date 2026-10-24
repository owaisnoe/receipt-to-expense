# Builds Theme3B_Receipt_to_Expense_FINAL.pptx (16:9) and its charts from the numbers measured on the Colab G4 run.
# Run: .venv\Scripts\python.exe build_deck.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qrcode
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)
OUT_PPTX = os.path.join(HERE, "Theme3B_Receipt_to_Expense_FINAL.pptx")

# ---------------------------------------------------------------- links (update APP_URL when the app is relaunched)
HF_URL = "https://huggingface.co/owaisnoe/gemma4-e2b-sroie-lora"
APP_URL = "https://f6753b53a276c78344.gradio.live"

# ---------------------------------------------------------------- palette (matches the app)
EMERALD, EMERALD_DARK, EMERALD_LIGHT, EMERALD_PALE = "#10b981", "#047857", "#6ee7b7", "#ecfdf5"
SLATE, SLATE_DARK, SLATE_MID, SLATE_LIGHT, SLATE_PALE = "#64748b", "#0f172a", "#334155", "#cbd5e1", "#f1f5f9"
AMBER, AMBER_PALE = "#f59e0b", "#fffbeb"
RED = "#dc2626"

def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

# ---------------------------------------------------------------- charts
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": SLATE_LIGHT, "axes.labelcolor": SLATE_MID, "xtick.color": SLATE_MID,
                     "ytick.color": SLATE_MID, "font.family": "DejaVu Sans"})

def _fmt(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

def bar_labels(ax, bars, fmt=None, size=12, color=SLATE_DARK):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.2, fmt.format(h) if fmt else _fmt(h), ha="center", va="bottom",
                fontsize=size, color=color)

def save(fig, name):
    p = os.path.join(CH, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

def chart_results():
    labels = ["Company", "Date", "Address", "Total", "All four\nnormalized", "All four\nstrict"]
    plain, prompted, ft = [76, 99, 52, 71, 28, 6], [85, 98, 63, 73, 37, 17], [92, 99, 72, 95, 66, 63]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    x, w = range(len(labels)), 0.27
    ax.axvspan(3.5, 5.5, color=EMERALD_PALE, zorder=0)
    b1 = ax.bar([i - w for i in x], plain, w, color=SLATE, label="Plain (short instruction)")
    b2 = ax.bar(list(x), prompted, w, color=AMBER, label="Prompted (detailed label rules)")
    b3 = ax.bar([i + w for i in x], ft, w, color=EMERALD, label="Fine-tuned (LoRA)")
    for bs in (b1, b2, b3):
        bar_labels(ax, bs, size=11)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% of 100 test receipts")
    ax.text(1.5, 106, "per field (normalized)", ha="center", fontsize=12, color=SLATE)
    ax.text(4.5, 106, "headline: all four fields right", ha="center", fontsize=12, color=EMERALD_DARK, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=12)
    return save(fig, "results.png")

def chart_seen_unseen():
    groups = ["Shops seen in training\n(63 receipts)", "New shops\n(37 receipts)"]
    plain, prompted, ft = [34.9, 16.2], [47.6, 18.9], [82.5, 37.8]
    fig, ax = plt.subplots(figsize=(5.9, 4.3))
    x, w = range(2), 0.26
    b1 = ax.bar([i - w for i in x], plain, w, color=SLATE, label="Plain")
    b2 = ax.bar(list(x), prompted, w, color=AMBER, label="Prompted")
    b3 = ax.bar([i + w for i in x], ft, w, color=EMERALD, label="Fine-tuned")
    for bs in (b1, b2, b3):
        bar_labels(ax, bs, size=12)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=13)
    ax.set_ylim(0, 100); ax.set_ylabel("all four fields right, % (normalized)")
    ax.set_title("SROIE test: seen vs new shops", fontsize=15, color=SLATE_DARK, loc="left")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    return save(fig, "seen_unseen.png")

def chart_ood():
    labels = ["all four", "total", "all fields", "total", "total"]
    plain, ft = [28, 71, 58, 87.5, 87.5], [66, 95, 56, 93.8, 91.7]
    pvals = ["p=5e-9", "p=3e-6", "p=1.0", "p=0.25", "p=0.62"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    x, w = range(len(labels)), 0.36
    ax.axvspan(1.5, 4.5, color=SLATE_PALE, zorder=0)
    b1 = ax.bar([i - w / 2 for i in x], plain, w, color=SLATE, label="Plain")
    b2 = ax.bar([i + w / 2 for i in x], ft, w, color=EMERALD, label="Fine-tuned")
    bar_labels(ax, b1, size=10.5); bar_labels(ax, b2, size=10.5)
    for i, p in enumerate(pvals):
        ax.text(i, max(plain[i], ft[i]) + 8, p, ha="center", fontsize=10.5, color=EMERALD_DARK if i < 2 else SLATE)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11.5)
    for gx, name in ((0.5, "SROIE (in-domain)"), (2.5, "WildReceipt"), (4, "CORD")):
        ax.text(gx, -19, name, ha="center", fontsize=12, weight="bold", color=SLATE_DARK, clip_on=False)
    ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% correct (labelled fields)")
    ax.text(3, 114, "never-seen receipt styles", ha="center", fontsize=11.5, color=SLATE_MID)
    ax.set_title("Other receipt datasets (50 each)", fontsize=15, color=SLATE_DARK, loc="left")
    ax.legend(frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return save(fig, "ood.png")

def chart_experiments():
    labels = ["Budget\n280", "Budget\n560 (main)", "Budget\n1120", "Rank\n8", "2\nepochs", "E4B\nmodel"]
    norm, strict = [61, 66, 76, 68, 71, 69], [56, 63, 70, 64, 67, 64]
    pv = ["", "", "p=0.006", "p=0.69", "p=0.18", "p=0.61"]
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    x, w = range(len(labels)), 0.38
    cols_n = [EMERALD] * 6; cols_n[2] = EMERALD_DARK
    b1 = ax.bar([i - w / 2 for i in x], norm, w, color=cols_n, label="all four, normalized")
    b2 = ax.bar([i + w / 2 for i in x], strict, w, color=EMERALD_LIGHT, label="all four, strict")
    bar_labels(ax, b1, size=12); bar_labels(ax, b2, size=11, color=SLATE_MID)
    ax.axhline(66, color=SLATE, lw=1.4, ls="--", zorder=0)
    ax.text(-0.62, 67.4, "main run 66%", ha="left", fontsize=10.5, color=SLATE)
    for i, p in enumerate(pv):
        if p:
            ax.text(i, 88, p + "\nvs main", ha="center", fontsize=10.5, color=EMERALD_DARK if i == 2 else SLATE)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100); ax.set_ylabel("fine-tuned, % of 100 test receipts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)
    return save(fig, "experiments.png")

def chart_learning_curve():
    steps, norm, strict = [0, 40, 80, 120, 157], [26, 54, 52, 54, 54], [6, 46, 48, 52, 52]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(steps, norm, "o-", color=EMERALD, lw=2.2, label="normalized")
    ax.plot(steps, strict, "s--", color=EMERALD_DARK, lw=1.6, label="strict")
    ax.set_xticks(steps); ax.set_ylim(0, 70); ax.set_yticks([0, 25, 50])
    ax.set_xlabel("training steps (x4 receipts)", fontsize=11); ax.set_ylabel("all four, %", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.annotate("most of the gain\nby 160 receipts", xy=(40, 54), xytext=(52, 12), fontsize=10, color=SLATE_MID,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.legend(frameon=False, fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.set_title("Learning curve (first 50 test receipts)", fontsize=11.5, color=SLATE_DARK, loc="left")
    return save(fig, "learning_curve.png")

def chart_rag():
    labels = ["Plain", "Prompted", "Fine-tuned"]
    before, after = [28, 37, 66], [39, 47, 69]
    pv = ["p=0.001", "p=0.002", "p=0.25"]
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    x, w = range(3), 0.36
    b1 = ax.bar([i - w / 2 for i in x], before, w, color=[SLATE, AMBER, EMERALD], label="without vendor memory")
    b2 = ax.bar([i + w / 2 for i in x], after, w, color=[SLATE, AMBER, EMERALD], alpha=0.55, hatch="//",
                edgecolor="white", label="+ vendor memory (RAG)")
    bar_labels(ax, b1, size=12); bar_labels(ax, b2, size=12)
    for i, p in enumerate(pv):
        ax.text(i, after[i] + 9, p, ha="center", fontsize=11, color=SLATE_MID)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 100); ax.set_ylabel("all four fields right, % (normalized)")
    ax.set_title("Vendor memory (RAG) on 100 SROIE receipts", fontsize=14, color=SLATE_DARK, loc="left")
    ax.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return save(fig, "rag.png")

def chart_lineitems():
    labels = ["Item F1", "Receipts with\nevery item right", "Total right"]
    plain, ft = [71.0, 54, 81.2], [78.8, 64, 93.8]
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    x, w = range(3), 0.36
    b1 = ax.bar([i - w / 2 for i in x], plain, w, color=SLATE, label="Plain")
    b2 = ax.bar([i + w / 2 for i in x], ft, w, color=EMERALD, label="Fine-tuned")
    bar_labels(ax, b1, size=12); bar_labels(ax, b2, size=12)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylim(0, 110); ax.set_ylabel("%")
    ax.set_title("Line items, zero-shot (50 CORD receipts, 126 lines)", fontsize=14, color=SLATE_DARK, loc="left")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    return save(fig, "lineitems.png")

def qr(url, name):
    p = os.path.join(CH, name)
    qrcode.make(url, box_size=10, border=2).save(p)
    return p

C = {"results": chart_results(), "seen": chart_seen_unseen(), "ood": chart_ood(), "exp": chart_experiments(),
     "lc": chart_learning_curve(), "rag": chart_rag(), "li": chart_lineitems(),
     "qr_hf": qr(HF_URL, "qr_hf.png"), "qr_app": qr(APP_URL, "qr_app.png")}

# ---------------------------------------------------------------- slide helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N_SLIDES = 13
_slide_no = [0]

def text(slide, x, y, w, h, runs, size=20, color=SLATE_DARK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.1, font="Calibri"):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run(); r.text = t
            f = r.font; f.name = font; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = rgb(o.get("color", color))
        if isinstance(para, list) and para and para[0][1].get("space_before"):
            p.space_before = Pt(para[0][1]["space_before"])
    return tb

def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s

def bullets(slide, x, y, w, h, items, size=20, color=SLATE_DARK, gap=6):
    paras = []
    for it in items:
        lvl2 = isinstance(it, tuple) and it[0] == 2
        t = it[1] if isinstance(it, tuple) else it
        if lvl2:
            paras.append([("      –  " + t, {"size": size - 2, "color": SLATE_MID, "space_before": 2})])
        else:
            paras.append([("•  ", {"color": EMERALD, "bold": True, "space_before": gap}), (t, {})])
    return text(slide, x, y, w, h, paras, size=size, color=color)

def new_slide(title, tag=None, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    _slide_no[0] += 1
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb("#ffffff")
    text(s, 0.6, 0.32, 11.0, 0.85, title, size=32, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.62, 1.17, 1.3, 0.07, EMERALD, shape=MSO_SHAPE.RECTANGLE)
    if subtitle:
        text(s, 2.05, 1.03, 10.6, 0.36, subtitle, size=15, color=SLATE)
    if tag:
        b = box(s, 11.45, 0.42, 1.3, 0.42, SLATE_PALE, line=SLATE_LIGHT, radius=0.3)
        text(s, 11.45, 0.42, 1.3, 0.42, tag, size=12, color=SLATE_MID, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 7.05, 9.0, 0.3, "Receipt to Expense  ·  Gemma 4 E2B + LoRA  ·  Theme 3B, Hacktoberfest Hack Day",
         size=11, color=SLATE)
    text(s, 11.9, 7.05, 0.85, 0.3, f"{_slide_no[0]} / {N_SLIDES}", size=11, color=SLATE, align=PP_ALIGN.RIGHT)
    return s

def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t

def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)

def stat(slide, x, y, w, h, big, label, color=EMERALD_DARK, fill=EMERALD_PALE, big_size=40, label_size=15):
    box(slide, x, y, w, h, fill)
    text(slide, x + 0.15, y + 0.08, w - 0.3, h * 0.55, big, size=big_size, bold=True, color=color,
         anchor=MSO_ANCHOR.BOTTOM)
    text(slide, x + 0.15, y + h * 0.58, w - 0.3, h * 0.4, label, size=label_size, color=SLATE_MID)

def table(slide, x, y, w, rows, col_w, size=14, header_fill=SLATE_DARK, row_h=0.36, first_col_bold=False,
          highlight_rows=(), highlight_fill=EMERALD_PALE):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.text = ""
            r = p.add_run(); r.text = str(val)
            r.font.name = "Calibri"; r.font.size = Pt(size)
            r.font.bold = (i == 0) or (first_col_bold and j == 0)
            r.font.color.rgb = rgb("#ffffff" if i == 0 else SLATE_DARK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if i == 0 else (highlight_fill if i in highlight_rows
                                                                          else ("#ffffff" if i % 2 else SLATE_PALE)))
    return tbl

def flow(slide, x, y, w, h, steps, fill=EMERALD_PALE, line=EMERALD, size=16, arrow=True):
    n = len(steps); gap = 0.32 if arrow else 0.15
    bw = (w - gap * (n - 1)) / n
    for i, s in enumerate(steps):
        bx = x + i * (bw + gap)
        box(slide, bx, y, bw, h, fill, line=line)
        text(slide, bx + 0.08, y + 0.05, bw - 0.16, h - 0.1, s, size=size, color=SLATE_DARK, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        if arrow and i < n - 1:
            a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(bx + bw + 0.05), Inches(y + h / 2 - 0.13),
                                       Inches(gap - 0.1), Inches(0.26))
            a.fill.solid(); a.fill.fore_color.rgb = rgb(EMERALD); a.line.fill.background()

# ================================================================ 1. Title (TALK 0:00)
s = prs.slides.add_slide(BLANK); _slide_no[0] += 1
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb("#ffffff")
box(s, 0, 0, 0.35, 7.5, EMERALD, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.9, 1.0, 11.5, 1.2, "Receipt to Expense", size=60, bold=True)
text(s, 0.9, 2.2, 12.1, 0.8, "A fine-tuned Gemma 4 E2B that reads receipts and splits the bill", size=26, color=SLATE_MID)
text(s, 0.9, 3.0, 11.5, 0.5, [[("Team Come_Join_Qarc", {"bold": True, "color": EMERALD_DARK}),
     ("  ·  SSIT Tumkur  ·  Theme 3B  ·  Hacktoberfest Hack Day  ·  24 Oct 2026", {})]], size=18, color=SLATE)
text(s, 0.9, 3.45, 11.5, 0.45, [[("Lead: ", {"bold": True}), ("Mohammed Owais   ", {}), ("Members: ", {"bold": True}),
     ("Mohammed Muzakkir · Mohammad Tariq Afzal · Mohammed Ayaan · Mohammed Younus", {})]], size=14, color=SLATE_MID)
stat(s, 0.9, 4.15, 3.0, 1.45, "66 %", "all four fields right\n(prompted Gemma: 37 %)")
stat(s, 4.1, 4.15, 3.0, 1.45, "2.5 min", "LoRA training on\none GPU (626 receipts)")
stat(s, 7.3, 4.15, 2.4, 1.45, "5/5 + 8/8", "deliverables +\nbonus items", big_size=30)
picture(s, C["qr_hf"], 10.55, 3.85, w=1.85)
text(s, 9.9, 5.75, 3.2, 0.9, [[("Adapters + model card", {"bold": True})], "huggingface.co/owaisnoe/\ngemma4-e2b-sroie-lora"],
     size=12, color=SLATE_MID, align=PP_ALIGN.CENTER)
text(s, 0.9, 6.55, 8.5, 0.4, "Gemma 4 E2B + LoRA (Unsloth)  ·  SROIE receipts  ·  Gradio app with a public link", size=14, color=SLATE)
notes(s, "TALK 0:00-0:12 (~38 words; opener story + hook)\n"
         "[Use a real story from the team if you have one; otherwise this example persona.]\n"
         "\"Riya's hostel flat splits groceries every week. The receipts end up in a WhatsApp photo pile, and someone types "
         "them into a spreadsheet. Our fine-tuned Gemma takes that from 6 to 63 percent of receipts read perfectly.\"\n"
         "(6 -> 63 % = all four fields exactly right, strict, plain vs fine-tuned, 100 SROIE test receipts.)\n\n"
         "Run order for the 3-minute talk: slides 1 > 2 > 3 > 4 > 5 (live demo) > 6 > 7. Slides 8-13 are backup for Q&A.\n"
         "Whole talk: about 365 spoken words, about 2:40 at 140 words/min plus demo actions. Rehearse with a timer.")

# ================================================================ 2. Problem and users (TALK 0:05)
s = new_slide("Problem: receipts are still typed in by hand", tag="SAAM · BHED")
text(s, 0.6, 1.45, 12.1, 0.95, [[("Who: ", {"bold": True, "color": EMERALD_DARK}),
     ("students sharing meals and trips, and small shops logging purchases.", {})],
     "Typing receipts in by hand is slow and error-prone."], size=20)
box(s, 0.6, 2.5, 5.85, 3.35, SLATE_PALE, line=SLATE_LIGHT)
text(s, 0.85, 2.62, 5.4, 0.5, "Today's expense apps", size=22, bold=True, color=SLATE_MID)
text(s, 0.85, 3.09, 5.4, 0.45, "built for companies, not for students", size=15, color=SLATE)
bullets(s, 0.85, 3.5, 5.4, 2.3, ["Closed, paid, cloud OCR", "Built for company cards", "Never say which field to trust"],
        size=20)
box(s, 6.85, 2.5, 5.85, 3.35, EMERALD_PALE, line=EMERALD)
text(s, 7.1, 2.62, 5.4, 0.5, "Receipt to Expense", size=22, bold=True, color=EMERALD_DARK)
text(s, 7.1, 3.09, 5.4, 0.45, "what we built today", size=15, color=SLATE)
bullets(s, 7.1, 3.5, 5.4, 2.3, ["Free, open 2B-class Gemma 4 E2B you can self-host + one LoRA file",
                                "Proven: baselines, significance tests, 3 datasets",
                                "Says which field to double-check, and splits the bill"], size=19)
box(s, 0.6, 6.0, 12.1, 0.85, AMBER_PALE, line=AMBER)
text(s, 0.8, 6.03, 11.75, 0.8, [[("No OCR step. ", {"bold": True, "color": AMBER}),
     ("Most receipt hacks chain OCR + an LLM; ours is one fine-tuned open vision-language model from photo to fields, "
      "measured with McNemar and out-of-distribution tests.", {})]], size=16, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "TALK 0:12-0:24 (~27 words). SAAM (relate) then BHED (set us apart)\n\"Every student has a receipt pile. Today's "
         "expense apps are paid, cloud-only, and never say which field to trust. Ours is a free, open model you can "
         "self-host, and it splits the bill.\"\n\n"
         "Pitch structure for the whole talk: SAAM = relate (the receipt pile, slide 2) · DAAM = the value we give (free, open, "
         "split the bill; the demo, slide 5) · DAND = the proof (6 -> 63 %, p = 1e-6; slides 4 and 6) · BHED = what sets us "
         "apart (no OCR step, a measured trust check; slide 2).")

# ================================================================ 3. Data and method (TALK 0:15)
s = new_slide("Data and method", subtitle="dataset description and training settings")
box(s, 0.6, 1.55, 5.95, 3.55, SLATE_PALE, line=SLATE_LIGHT)
text(s, 0.85, 1.65, 5.5, 0.5, "Data: SROIE (ICDAR 2019, CC BY 4.0)", size=21, bold=True, color=SLATE_MID)
bullets(s, 0.85, 2.12, 5.55, 2.95, ["626 train / 361 test scanned receipts", "Labels: company, date, address, total",
                                    "Test set: fixed 100 receipts (seed 0); keys asserted disjoint from training",
                                    "Never trained on test, external or local receipts"], size=18, gap=4)
box(s, 6.8, 1.55, 5.95, 3.55, EMERALD_PALE, line=EMERALD)
text(s, 7.05, 1.65, 5.5, 0.5, "Method: LoRA on Gemma 4 E2B", size=21, bold=True, color=EMERALD_DARK)
bullets(s, 7.05, 2.12, 5.55, 2.95, ["LoRA r = 32 on vision + language layers: 59.8 M trainable (1.15 %)",
                                    "5 smoke + 157 steps = 1 epoch: 2.5 min on one G4 GPU, peak 13 GB",
                                    "Image first, 560 visual tokens (525-544 measured)",
                                    "Greedy decoding, batch size 1, for every configuration"], size=18, gap=4)
flow(s, 0.6, 5.4, 12.15, 1.15, ["Receipt photo", "Gemma 4 E2B + LoRA\nimage first, 560 tokens",
                                "JSON: company, date,\naddress, total", "App: check, split,\nadd to expenses"], size=16)
notes(s, "TALK 0:24-0:32 (~18 words)\n\"Data: SROIE, 626 training and 361 test receipts; we test on a fixed 100 we "
         "never trained on.\"\n\n"
         "Compute (if asked): trained on a Colab G4 for speed (2.5 min). Unsloth's docs put E2B LoRA at about 8-10 GB, so the "
         "recipe should also fit a free T4, more slowly; we did not test this.")

# ================================================================ 4. The one chart (TALK 0:25-1:10)
s = new_slide("Fine-tuning beats prompting on every field", tag="DAND")
picture(s, C["results"], 0.5, 1.45, h=4.95)
stat(s, 9.3, 1.55, 3.45, 1.55, "66 %", "all four fields, normalized\nprompted 37 %  ·  plain 28 %")
stat(s, 9.3, 3.25, 3.45, 1.55, "63 %", "all four fields, strict\nprompted 17 %  ·  plain 6 %")
stat(s, 9.3, 4.95, 3.45, 1.55, "p = 1e-6", "McNemar vs prompted:\n33 receipts better, 4 worse", big_size=34)
text(s, 0.6, 6.45, 12.1, 0.58, "Same 100 test receipts, same 560-token image budget, greedy batch 1. Normalized forgives case, "
     "spacing, currency and date format, so it measures reading, not formatting.", size=13, color=SLATE)
notes(s, "TALK 0:32-1:12 (~85 words)\n\"One chart: same 100 receipts, same image budget, greedy decoding. Plain Gemma with a "
         "short instruction; the same model with a detailed prompt that spells out the label rules; and our LoRA fine-tune. "
         "Normalized scoring forgives case, currency and date format, so it measures reading, not formatting. All four fields "
         "right: plain 28 percent, prompted 37, fine-tuned 66. Against the prompted baseline, McNemar gives p of about one in "
         "a million: not just formatting. The biggest jump is the total, 73 to 95.\"\n\n"
         "Exact values: McNemar fine-tuned vs prompted, normalized 33 better / 4 worse, p = 1.1e-6; strict 48 / 2, p = 2.3e-12. "
         "Vs plain: normalized 42 / 4, p = 5.1e-9; strict 60 / 3, p = 9e-15.")

# ================================================================ 5. The app (TALK 1:10-2:20)
s = new_slide("Live demo: snap a receipt, check it, split it", tag="DAAM",
              subtitle="working application built around the fine-tuned model")
steps = [("1", "Snap or upload", "phone camera, webcam or a file"),
         ("2", "Fine-tuned fields", "editable; plain model beside it; trust check: \"agree on N/4 fields, double-check X\""),
         ("3", "Known-vendor badge", "vendor memory from the training labels (RAG)"),
         ("4", "Split the bill", "equally or item by item; items auto-filled by zero-shot line items; shared by names or 'all'"),
         ("5", "Group balances", "tax, service and rounding shared in proportion; settle up with the fewest payments"),
         ("6", "Expenses + CSV", "running total of expenses; download the ledger as CSV")]
for i, (num, head, sub) in enumerate(steps):
    col, row = i % 2, i // 2
    split = num in ("4", "5")
    bx, by = 0.6 + col * 4.55, 1.55 + row * 1.6
    box(s, bx, by, 4.35, 1.45, AMBER_PALE if split else EMERALD_PALE, line=AMBER if split else EMERALD)
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(bx + 0.18), Inches(by + 0.4), Inches(0.62), Inches(0.62))
    c.fill.solid(); c.fill.fore_color.rgb = rgb(AMBER if split else EMERALD); c.line.fill.background()
    text(s, bx + 0.18, by + 0.4, 0.62, 0.62, num, size=20, bold=True, color="#ffffff", align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, bx + 0.95, by + 0.1, 3.3, 0.45, head, size=20, bold=True)
    text(s, bx + 0.95, by + 0.55, 3.3, 0.88, sub, size=14, color=SLATE_MID)
box(s, 9.85, 1.55, 2.9, 4.65, SLATE_PALE, line=SLATE_LIGHT)
text(s, 9.95, 1.63, 2.7, 0.45, "Try it on your phone", size=17, bold=True, align=PP_ALIGN.CENTER)
picture(s, C["qr_app"], 10.25, 2.1, w=2.1)
text(s, 9.95, 4.25, 2.7, 1.9, [[(APP_URL.replace("https://", ""), {"size": 11.5})], "",
     [("Judges: lend us a receipt!", {"bold": True, "color": AMBER})],
     "Extract about 7 s (fine-tuned + plain); \"please check\" when a date or total does not parse"], size=12,
     color=SLATE_MID, align=PP_ALIGN.CENTER)
text(s, 0.6, 6.33, 12.15, 0.68, [[("Open model: ", {"bold": True, "color": EMERALD_DARK}),
     ("Gemma 4 E2B (2B-class) + one LoRA adapter file: no cloud OCR, no subscription.", {})],
     [("Built for students: ", {"bold": True, "color": AMBER}),
      ("split equally or by item, tax shared fairly, everyone settled in the fewest payments.", {})]],
     size=13, color=SLATE_MID)
notes(s, "TALK 1:12-2:20 (~100 words + actions)\n\"Now yours: could one of you lend me a receipt?\" [photograph it, press Extract]\n"
         "\"This runs on a Colab GPU behind a public link; scan the QR to try it on your phone. The fine-tuned model fills the "
         "form, with the plain model beside it; when they disagree, it tells you which field to double-check. If the shop is "
         "in our vendor memory, it says so. Then split the bill: equally or item by item, with the items read off the "
         "receipt; tax and service are shared in proportion, and it settles up with the fewest payments. Add to expenses "
         "keeps a running total and a CSV.\"\n\n"
         "Trust check (measured on 100 test receipts, see the backup slide): when both models agree on all four fields "
         "(29 % of receipts) the fine-tuned answer is fully right 83 % of the time, vs 59 % otherwise; it flags 75 % of "
         "company and 64 % of address errors but only 40 % of total errors. Extract takes about 7 s (both models).\n"
         "Split details: 'shared by' takes names or 'all'; tax, service and rounding go into an 'Others' row shared in "
         "proportion, with a 'make sure this amount is correct' warning when it is large; a 'Left to split 0.00' check; the "
         "group ledger settles up with the fewest payments (e.g. 'Arjun pays Me 40.00') and downloads as CSV.\n"
         "Fallbacks: if a judge's receipt reads badly, say what failed and click a built-in example; if the link is dead, "
         "play the backup video. App link: " + APP_URL)

# ================================================================ 6. Where it helped and where it didn't (TALK 2:20-2:45)
s = new_slide("Wins on new shops too, and stays safe on other receipts", tag="DAND",
              subtitle="where fine-tuning helps, its limits, and why")
picture(s, C["seen"], 0.55, 1.42, h=4.4)
picture(s, C["ood"], 6.95, 1.42, h=4.4)
box(s, 0.6, 5.95, 12.15, 0.95, EMERALD_PALE, line=EMERALD)
text(s, 0.8, 6.0, 11.8, 0.85, [[("Helps most: ", {"bold": True, "color": EMERALD_DARK}),
     ("totals (73 -> 95 %), addresses and receipt conventions; on brand-new shops it still doubles prompting (38 vs 19 %).  ", {}),
     ("Next frontier: ", {"bold": True, "color": AMBER}),
     ("other countries' receipts, safe today (totals up on all 3 datasets); Indian receipts next.", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "TALK 2:20-2:45 (~60 words). DAND (proof that it generalises)\n\"It wins on shops it has never seen: 38 percent "
         "versus 19 for the prompted model. On other countries' receipts it stays safe: totals improve on all three datasets. "
         "It was trained on Malaysian receipts, so Indian receipts are our next step. It even catches label errors: the gold "
         "label says B1750, the receipt says 81750.\"\n\n"
         "Numbers: seen vs new shops, all four normalized: fine-tuned 82.5 vs 37.8; prompted 47.6 vs 18.9; plain 34.9 vs 16.2. "
         "WildReceipt all labelled fields 58 -> 56 (p = 1.0); total 87.5 -> 93.8 (p = 0.25). CORD total 87.5 -> 91.7 (p = 0.62). "
         "Total-field change from fine-tuning: SROIE +24, WildReceipt +6.3, CORD +4.2.\n\n"
         "If asked for prior evidence: our research brief cites arXiv 2609.15706 as finding that fine-tuned small open VLMs "
         "beat zero-shot commercial systems on templated extraction. We could not re-verify it today; check before quoting.")

# ================================================================ 7. Scorecard and next steps (TALK 2:45-3:00)
s = new_slide("Delivered: 5 / 5 required, 8 / 8 bonus", tag="DELIVERED")
rows = [["Bonus item", "Status", "Evidence"],
        ["Prompted baseline", "Done", "37 % vs fine-tuned 66 % (p = 1e-6)"],
        ["Training experiments", "Done", "budget 280 / 560 / 1120, rank 8, 2 epochs, curve"],
        ["Overfitting check", "Done", "WildReceipt + CORD: no loss; wins on new shops"],
        ["Model size (E4B)", "Done", "69 vs 66 % (n.s.); 3.46 vs 3.0 s; 17.4 vs ~11 GB"],
        ["Combined with RAG", "Done", "vendor memory: 66 -> 69 %, prompted 37 -> 47 %"],
        ["Publishing on HF", "Done", "adapters + model card on Hugging Face"],
        ["Error analysis", "Done", "9 categories, fixes proposed"],
        ["3B: line items", "Done (zero-shot)", "CORD item F1 71 -> 79 %"]]
table(s, 0.6, 1.5, 7.6, rows, [2.35, 1.6, 3.65], size=14, row_h=0.5)
box(s, 8.5, 1.5, 4.25, 2.3, EMERALD_PALE, line=EMERALD)
text(s, 8.7, 1.58, 3.9, 0.45, "Required: 5 / 5 ✓", size=20, bold=True, color=EMERALD_DARK)
bullets(s, 8.7, 1.98, 3.95, 1.8, ["Fine-tuned model + settings (HF)", "Dataset description (slide 3)",
                                  "Comparison + examples (slide 4)", "Working app, live (slide 5)",
                                  "Helps / limits / why (slide 6)"], size=14, gap=1)
box(s, 8.5, 3.95, 4.25, 2.55, SLATE_PALE, line=SLATE_LIGHT)
text(s, 8.7, 4.03, 3.9, 0.45, "Next", size=20, bold=True, color=SLATE_MID)
bullets(s, 8.7, 4.43, 3.95, 2.05, ["Retrain at 1120 visual tokens (76 %)", "Indian receipts + local photos",
                                   "UPI payment requests from the settle-up list",
                                   "On-device 4-bit E2B: receipts stay on the phone",
                                   "Calibrated trust score (totals)"], size=14, gap=1)
notes(s, "TALK 2:45-3:00 (~38 words)\n\"All eight bonus items are done. The biggest lever was resolution: 1120 visual tokens reach "
         "76 percent, more than model size or rank. Next: Indian receipts, on-device, and a better trust score. Adapters and "
         "model card are on Hugging Face.\"\n\nHF: " + HF_URL)

# ================================================================ 8. Training experiments (BACKUP)
s = new_slide("What moves accuracy: resolution, not size", tag="BACKUP", subtitle="training experiments + model size, same 100 test receipts")
picture(s, C["exp"], 0.45, 1.45, h=4.95)
box(s, 8.15, 1.45, 4.6, 2.05, EMERALD_PALE, line=EMERALD)
bullets(s, 8.3, 1.47, 4.4, 2.0, ["1120 tokens: 76 % vs 66 % (p = 0.006)",
                                 "Rank 8 = rank 32 (68 vs 66 %, n.s.)", "2 epochs: 71 % (n.s.)",
                                 "E4B: 69 % (n.s.); 3.46 vs 3.0 s per receipt, 17.4 vs ~11 GB peak"], size=15, gap=3)
picture(s, C["lc"], 8.6, 3.65, h=2.7)
text(s, 0.6, 6.42, 12.1, 0.6, "Each budget arm is trained and tested at its own budget. Plain gets worse with more visual tokens "
     "(33 -> 28 -> 22 %) while fine-tuned gets better (61 -> 66 -> 76 %): an observation, not yet explained.", size=13, color=SLATE)
notes(s, "BACKUP (Q&A): training experiments and the E4B size comparison.\n"
         "All on the same 100 SROIE test receipts, all-four normalized / strict, exact McNemar vs the main fine-tuned E2B "
         "(r32, 157 steps, budget 560: 66 / 63).\n"
         "- Visual budget: 280 -> 61 / 56; 560 -> 66 / 63; 1120 -> 76 / 70 (+11 / -1 vs 560, p = 0.0063). Plain: 33 / 10, 28 / 6, 22 / 7.\n"
         "- LoRA rank 8: 68 / 64 (+4 / -2, p = 0.69): rank 8 is enough.\n"
         "- 2 epochs (314 steps): 71 / 67 (+7 / -2, p = 0.18, n.s.). The 1-epoch learning curve plateaus by step 40 "
         "(first 50 receipts: 26 / 6 -> 54 / 46 at step 40 -> 54 / 52 at 157).\n"
         "- E4B (r32, 157 steps, 560): plain 42 / 16, prompted 44 / 32, fine-tuned 69 / 64; vs fine-tuned E2B +9 / -6, p = 0.61 "
         "(n.s.); vs its own prompted baseline +28 / -3, p = 4.6e-6. Clean inference speed (idle GPU, 20 receipts): E4B "
         "fine-tuned 3.46 s/receipt and 17.4 GB peak vs E2B fine-tuned 3.0 s/receipt and ~11 GB. Training peak 17.9 GB (E4B) "
         "vs 13.0 GB (E2B); training 6.35 min (2.43 s/step, GPU shared) vs 2.5 min (0.96 s/step, idle GPU).\n"
         "Message: resolution matters more than model size, rank or epochs; 1120 is the recommended next configuration "
         "(the demo app uses the main 560 model).")

# ================================================================ 9. Trust check (BACKUP)
s = new_slide("Trust check: which field to double-check", tag="BACKUP",
              subtitle="fine-tuned vs plain agreement, measured on the 100 SROIE test receipts")
rows = [["Field", "Models agree", "Fine-tuned right\nwhen they agree", "Fine-tuned right\nwhen they differ",
         "Errors flagged\nby \"differ\""],
        ["Company", "76 %", "97 %", "75 %", "75 %"], ["Date", "99 %", "100 %", "0 %", "100 %"],
        ["Address", "59 %", "83 %", "56 %", "64 %"], ["Total", "72 %", "96 %", "93 %", "40 %"]]
table(s, 0.6, 1.6, 8.2, rows, [1.6, 1.5, 1.7, 1.7, 1.7], size=16, row_h=0.72, first_col_bold=True)
stat(s, 9.15, 1.6, 3.6, 1.65, "29 %", "of receipts: all four fields agree")
stat(s, 9.15, 3.4, 3.6, 1.65, "83 % vs 59 %", "all four right when they agree\nvs when any field differs", big_size=30)
box(s, 0.6, 5.45, 12.15, 1.45, AMBER_PALE, line=AMBER)
text(s, 0.8, 5.5, 11.8, 1.35, [[("Useful but imperfect. ", {"bold": True, "color": AMBER}),
     ("A disagreement flags 3 in 4 company errors and about 2 in 3 address errors, but only 40 % of total errors: "
      "both models can agree on the same wrong total. The app shows it as \"agree on N/4 fields; double-check X\".", {})]],
     size=17, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "BACKUP (Q&A): the trust check shown in the app.\n"
         "For each field we compare the fine-tuned and plain answers (normalized). On the 100 test receipts: company agrees on "
         "76 % of receipts (fine-tuned right 97 % when they agree, 75 % when they differ; 75 % of fine-tuned company errors are "
         "flagged by a disagreement); date 99 % / 100 % / 0 % / 100 %; address 59 % / 83 % / 56 % / 64 %; total 72 % / 96 % / "
         "93 % / 40 %. Receipt level: all four fields agree on 29 % of receipts, which are 83 % all-four right vs 59 % when any "
         "field differs. Present it honestly: a cheap, useful signal, weakest for totals. A calibrated score is a next step.")

# ================================================================ 10. Error analysis (BACKUP)
s = new_slide("Error analysis: most misses are one-character slips", tag="BACKUP", subtitle="fine-tuned model, 100 test receipts")
rows = [["Category", "Company", "Date", "Address", "Total"],
        ["correct", 91, 99, 71, 92], ["format only", 1, 0, 1, 3], ["punctuation / spacing", 3, 0, 9, 0],
        ["near miss (1-2 chars)", 2, 0, 11, 0], ["OCR-confusable (8/B)", 0, 0, 3, 0], ["extra text", 2, 0, 1, 0],
        ["truncated", 0, 0, 1, 0], ["rounding (<= 0.05)", 0, 0, 0, 3], ["different value", 1, 1, 3, 2]]
table(s, 0.6, 1.55, 6.3, rows, [2.5, 0.95, 0.85, 1.05, 0.95], size=14, row_h=0.43, first_col_bold=True, highlight_rows=(1,))
ex = [("Address", "gold  ...B1750 MASAI", "read  ...81750 MASAI", "label noise: 81750 is Masai's real postcode"),
      ("Total", "gold  7.42", "read  RM7.40", "rounding adjustment printed on the receipt"),
      ("Total", "gold  6.40", "read  6.04", "digit swap"),
      ("Company", "gold  THONG RECIPE", "read  16G BLOCK B, ZENITH CORPORATE PARK", "read the address line as the name")]
for i, (f, g, r, why) in enumerate(ex):
    by = 1.55 + i * 1.18
    box(s, 7.2, by, 5.55, 1.05, SLATE_PALE, line=SLATE_LIGHT)
    text(s, 7.35, by + 0.04, 5.3, 1.0, [[(f + ":  ", {"bold": True, "color": EMERALD_DARK}), (g, {"color": SLATE_MID})],
                                         [("read  ", {"color": SLATE_MID}), (r.replace("read  ", ""), {"bold": True})],
                                         [(why, {"italic": True, "size": 13, "color": SLATE})]], size=14.5, spacing=1.0)
text(s, 0.6, 6.45, 12.1, 0.5, [[("Fixes: ", {"bold": True, "color": EMERALD_DARK}),
     ("punctuation-tolerant address matching  ·  vendor memory (RAG)  ·  higher visual budget (1120: 76 %)", {})]], size=16)
notes(s, "BACKUP (Q&A): error analysis of the fine-tuned model (section 13b categories, one per field).\n"
         "Address is the weak field: of 28 wrong addresses, 9 are punctuation or spacing only, 11 are near misses of one or two "
         "characters (e.g. KEWASAN vs KAWASAN, 61200 vs 81200), and 3 involve OCR-confusable characters, including the gold-label "
         "error B1750 (both models read 81750, the real postcode of Masai, Johor). Totals improved the most: the plain model had "
         "13 missing and 13 wrong amounts; the fine-tuned model has 2 different amounts (one digit swap, 6.04 vs 6.40) and 3 "
         "Malaysian rounding adjustments (7.40 vs 7.42). Date: 1 truncated (11/16 vs 11/11/16).")

# ================================================================ 10. Engineering findings (BACKUP)
s = new_slide("What we found under the hood", tag="BACKUP", subtitle="understanding how it works: a hidden bug caught and fixed")
cards = [("A silent resolution bug (fixed)",
          ["Unsloth's collator shrank receipts to 512 px and snapped them to 32 px",
           "Token count identical (525 = 525), pixels not: max difference 0.68",
           "Fix: images column + resize=\"max\": difference 0.0000"]),
         ("Known SDPA bug: checked",
          ["torch <= 2.13 corrupts prompts of length 1 mod 32",
           "0 / 100 of our prompts affected (both instructions)",
           "Eager attention matched SDPA exactly on 3 + 3 receipts"]),
         ("80 of 718 LoRA tensors never trained",
          ["They stayed identical across every checkpoint",
           "Our explanation (not verified): k/v projections in E2B's KV-shared layers get no gradient",
           "Smoke run said 12.5 s/step (warm-up); real 0.96 s/step"])]
for i, (head, items) in enumerate(cards):
    bx = 0.6 + i * 4.1
    box(s, bx, 1.6, 3.9, 4.75, EMERALD_PALE if i == 0 else SLATE_PALE, line=EMERALD if i == 0 else SLATE_LIGHT)
    text(s, bx + 0.2, 1.72, 3.5, 0.95, head, size=20, bold=True, color=EMERALD_DARK if i == 0 else SLATE_MID)
    bullets(s, bx + 0.2, 2.62, 3.55, 3.65, items, size=16, gap=6)
text(s, 0.6, 6.42, 12.1, 0.6, "Gates passed before every run: budget applied (525-544 image tokens), pixel parity, labels contain "
     "the JSON answer, finite losses (first loss 2.56).", size=13, color=SLATE)
notes(s, "BACKUP (Q&A): engineering findings.\n"
         "1) The collator bug: training and inference showed the same image-token count, 525 vs 525, but Unsloth's "
         "UnslothVisionDataCollator by default resizes images to 512 px wide (resize='min', the model config has no default size) "
         "and, for images inside the messages, snaps sizes to multiples of 32 px. Max pixel difference 0.34 on a 463x1013 receipt "
         "and 0.68 on a 4961x7016 receipt. Fix: an images column plus resize='max' -> identical pixel_values (0.0000) on six "
         "receipts. Without it, the model would train on blurrier receipts than it reads at test time.\n"
         "2) SDPA prefill bug (torch <= 2.13, prompt length = 1 mod 32): lengths were 541-587 and 663-709, none affected; eager "
         "matched SDPA on 3 + 3 receipts.\n"
         "3) 80 of 718 LoRA tensors were identical in every checkpoint; our explanation (not verified) is the KV-shared layers.\n"
         "Also: the smoke run's 12.5 s/step included compile warm-up; 157 steps took 2.5 min (0.96 s/step).")

# ================================================================ 11. RAG + line items (BACKUP)
s = new_slide("Bonus: vendor memory (RAG) and line items", tag="BACKUP")
picture(s, C["rag"], 0.55, 1.45, h=4.25)
picture(s, C["li"], 6.75, 1.45, h=4.25)
text(s, 0.6, 5.85, 5.95, 1.1, "276 shops from the training labels only; match threshold 0.85 fixed in advance. Fine-tuning had "
     "already learned most of what retrieval adds.", size=15, color=SLATE_MID)
text(s, 6.9, 5.85, 5.85, 1.1, "Zero-shot: SROIE has no line-item labels, so we scored on CORD. No forgetting; the fine-tuned "
     "model writes names in capitals and once shifted prices by a row.", size=15, color=SLATE_MID)
notes(s, "BACKUP (Q&A): RAG and line items.\n"
         "RAG = retrieval-augmented post-correction: the predicted company and address are matched against a vendor memory of "
         "276 (company, address) pairs from 625 TRAIN receipts (test labels never used; threshold 0.85 fixed a priori). "
         "All four normalized: plain 28 -> 39 (11 better / 0 worse, p = 0.001), prompted 37 -> 47 (10 / 0, p = 0.002), fine-tuned "
         "66 -> 69 (3 / 0, p = 0.25, n.s.).\n"
         "Line items, zero-shot on 50 CORD receipts (126 gold item lines): item precision 62.8 vs 75.4, recall 81.7 vs 82.5, "
         "F1 71.0 vs 78.8; receipts with every item right 54 vs 64 % (10 / 5, p = 0.3); total right 81.2 vs 93.8 %; quantities "
         "right on all matched items. Failure example: on one long receipt the fine-tuned model dropped a line and shifted every "
         "later price up by one row.")

# ================================================================ 12. Appendix: settings (BACKUP)
s = new_slide("Appendix: settings and reproducibility", tag="APPENDIX")
rows = [["Setting", "Value"],
        ["Base model / load", "unsloth/gemma-4-E2B-it, 16-bit LoRA (load_in_4bit = False)"],
        ["LoRA", "r 32, alpha 32, dropout 0; vision + language, attention + MLP; 59.8 M / 5.18 B (1.15 %)"],
        ["Optimiser", "adamw_8bit, lr 2e-4, cosine, warm-up 0.03, weight decay 0.001, grad norm 0.3"],
        ["Batch / steps", "1 x grad-accum 4; 5 smoke + 157 steps (1 epoch over 626 receipts)"],
        ["Visual budget / prompt", "max_soft_tokens 560 (525-544 tokens); image first, then instruction"],
        ["Instruction", "Extract the company, date, address and total from this receipt. Reply with JSON only."],
        ["Decoding", "greedy, batch size 1, max_new_tokens 256 (every configuration)"],
        ["Test set", "100 of 361 SROIE test receipts, random.Random(0); keys in test_keys.json"],
        ["Hardware / time", "RTX PRO 6000 Blackwell (Colab G4, 95 GB); 2.5 min training, peak 13.0 GB"],
        ["Speed (idle GPU)", "plain 2.4 s, prompted 2.0 s, fine-tuned 3.0 s per receipt"],
        ["Software", "torch 2.11 cu130, transformers 5.5.0, unsloth 2026.10.8, trl 1.16.0, peft 0.21.2"]]
table(s, 0.6, 1.5, 12.15, rows, [2.9, 9.25], size=14, row_h=0.4, first_col_bold=True)
text(s, 0.6, 6.38, 12.15, 0.6, [[("Compute: ", {"bold": True, "color": EMERALD_DARK}),
     ("trained on a Colab G4 for speed (2.5 min). Unsloth's docs put E2B LoRA at about 8-10 GB, so the recipe should also "
      "fit a free T4, more slowly; we did not test this.", {})]], size=13, color=SLATE_MID)
notes(s, "APPENDIX: the training-settings deliverable (also saved as summary.json in the notebook and in the model card).\n"
         "Gates on the runtime: B (560 budget applied: 525-544 image tokens), B2 (training collator pixels = inference pixels, "
         "max difference 0.0000), C (the trained text contains the JSON answer; image tokens masked; the instruction is also "
         "trained, the Unsloth default), D (finite losses: smoke 2.56 -> 1.16; main run 0.98 at step 5, 0.016 at step 155). "
         "Adapters saved with save_pretrained, never merged. HF: " + HF_URL)

assert _slide_no[0] == N_SLIDES, _slide_no[0]
prs.save(OUT_PPTX)
print("saved", OUT_PPTX, "|", _slide_no[0], "slides")
