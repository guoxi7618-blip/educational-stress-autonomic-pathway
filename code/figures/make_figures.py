"""Build Figs. 1-6 from the Source Data file.

Every figure is defined once as a scene in inches. Schematics become native,
editable PowerPoint shapes and data panels become Matplotlib panels, so the
PowerPoint deck and the PDF/PNG exports share one layout.

Usage: python make_figures.py
Outputs: Fig1-6 .pdf/.png and Figures_editable.pptx, plus one PNG per data panel for the deck.
In the public repository they go to figures/.
"""
import os
import re
import csv
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUB = os.path.join(ROOT, "投稿稿件")
SOURCE = os.path.join(SUB, "第四篇终稿", "06_Source_Data.xlsx")
OUT = os.path.join(SUB, "第四篇终稿", "04_Figures")
PANELS = os.path.join(SUB, "figures", "panels")
if not os.path.exists(SOURCE):  # layout of the public repository
    SOURCE = os.path.join(ROOT, "source_data", "Source_Data.xlsx")
    OUT = os.path.join(ROOT, "figures")
    PANELS = os.path.join(ROOT, "figures", "panels")
MULTILEVEL = os.path.join(ROOT, "data_for_reviewers", "04_daily_bursts_and_devices", "daily_multilevel_models.csv")
AGREEMENT = os.path.join(ROOT, "data_for_reviewers", "04_daily_bursts_and_devices", "device_agreement.csv")

WIDTH = 7.087  # 180 mm, journal double-column width
DPI = 600

BLUE = "#1F6FB2"
ORANGE = "#D9622B"
INK = "#222222"
GREY = "#595959"
MID = "#9A9A9A"
LIGHT = "#BDBDBD"
PALE = "#EFEFEF"
T_STRESS = "#FBE6DC"
T_AUTO = "#DDEAF6"
T_CRISIS = "#DFF1EA"
PROFILE = {"P1": "#009E73", "P2": "#1F6FB2", "P3": "#808080", "P4": "#7A5AB5", "P5": "#D9622B"}
STATE = ["#D3E2F2", "#8FB8E0", "#3F86C6", "#1B5A96", "#0B2F57"]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
})


# ---------------------------------------------------------------- data
BOOK = openpyxl.load_workbook(SOURCE, data_only=True)


def sheet(name):
    rows = list(BOOK[name].iter_rows(values_only=True))
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def ci(text):
    lo, hi = re.findall(r"-?\d+\.?\d*", text.replace("−", "-"))[-2:]
    return float(lo), float(hi)


def num(v, digits=2):
    """Round half up, as in the manuscript, and use a true minus sign."""
    q = Decimal(1).scaleb(-digits)
    return f"{Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP)}".replace("-", "−")


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


# ---------------------------------------------------------------- scene
class Scene:
    """Shapes and panels with coordinates in inches from the top-left corner."""

    def __init__(self, name, height):
        self.name, self.height, self.items = name, height, []

    def letter(self, s, x, y):
        self.items.append(("text", dict(x=x, y=y, s=s, size=8, bold=True, ha="left", va="top", color=INK)))

    def text(self, x, y, s, size=6, ha="left", va="top", color=INK, bold=False):
        self.items.append(("text", dict(x=x, y=y, s=s, size=size, bold=bold, ha=ha, va=va, color=color)))

    def box(self, x, y, w, h, s="", fill=PALE, color=INK, size=6.5, bold=False, round_=True, bold_first=False):
        self.items.append(("box", dict(x=x, y=y, w=w, h=h, s=s, fill=fill, color=color, size=size,
                                       bold=bold, round_=round_, bold_first=bold_first)))

    def arrow(self, x1, y1, x2, y2, color=GREY, lw=0.8, dash=False, head=True):
        self.items.append(("arrow", dict(x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw, dash=dash, head=head)))

    def panel(self, key, x, y, w, h, l, t, r, b, fn):
        """Outer panel region (x, y, w, h); axes inset by l, t, r, b inches."""
        self.items.append(("panel", dict(key=key, x=x, y=y, w=w, h=h, l=l, t=t, r=r, b=b, fn=fn)))


def add_axes(fig, W, H, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def render_mpl(scene):
    W, H = WIDTH, scene.height
    fig = plt.figure(figsize=(W, H))
    base = fig.add_axes([0, 0, 1, 1])
    base.set_xlim(0, W); base.set_ylim(H, 0); base.axis("off")
    for z, (kind, d) in enumerate(scene.items, start=2):
        if kind == "box":
            if d["round_"]:
                p = FancyBboxPatch((d["x"], d["y"]), d["w"], d["h"], boxstyle="round,pad=0,rounding_size=0.035",
                                   fc=d["fill"], ec="none", lw=0, mutation_aspect=1, zorder=z)
            else:
                p = Rectangle((d["x"], d["y"]), d["w"], d["h"], fc=d["fill"], ec="none", lw=0, zorder=z)
            base.add_patch(p)
            if d["s"]:
                lines = d["s"].split("\n")
                n = len(lines)
                step = d["size"] * 1.2 / 72
                for i, line in enumerate(lines):
                    yy = d["y"] + d["h"] / 2 + (i - (n - 1) / 2) * step
                    base.text(d["x"] + d["w"] / 2, yy, line, ha="center", va="center", fontsize=d["size"],
                              color=d["color"], zorder=z + 0.5,
                              fontweight="bold" if d["bold"] or (d["bold_first"] and i == 0) else "normal")
        elif kind == "text":
            base.text(d["x"], d["y"], d["s"], fontsize=d["size"], ha=d["ha"], va=d["va"], color=d["color"],
                      fontweight="bold" if d["bold"] else "normal", linespacing=1.2, zorder=z)
        elif kind == "arrow":
            style = "-|>,head_width=0.11,head_length=0.2" if d["head"] else "-"
            base.annotate("", xy=(d["x2"], d["y2"]), xytext=(d["x1"], d["y1"]),
                          arrowprops=dict(arrowstyle=style, color=d["color"], lw=d["lw"], shrinkA=0, shrinkB=0,
                                          linestyle=(0, (3, 2)) if d["dash"] else "-", mutation_scale=8), zorder=z)
        elif kind == "panel":
            ax = add_axes(fig, W, H, d["x"] + d["l"], d["y"] + d["t"], d["w"] - d["l"] - d["r"], d["h"] - d["t"] - d["b"])
            d["fn"](ax)
    fig.savefig(os.path.join(OUT, scene.name + ".pdf"))
    fig.savefig(os.path.join(OUT, scene.name + ".png"), dpi=DPI)
    plt.close(fig)


def rgb(h):
    return RGBColor.from_string("FFFFFF" if h == "white" else h.lstrip("#"))


def set_runs(tf, s, size, color, bold, bold_first=False, align=PP_ALIGN.CENTER):
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.name = "Arial"; r.font.color.rgb = rgb(color)
        r.font.bold = bool(bold or (bold_first and i == 0))


def render_ppt(prs, scene):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for kind, d in scene.items:
        if kind == "box":
            shape = MSO_SHAPE.ROUNDED_RECTANGLE if d["round_"] else MSO_SHAPE.RECTANGLE
            sp = slide.shapes.add_shape(shape, Inches(d["x"]), Inches(d["y"]), Inches(d["w"]), Inches(d["h"]))
            if d["round_"]:
                sp.adjustments[0] = min(0.035 / min(d["w"], d["h"]), 0.5)
            sp.fill.solid(); sp.fill.fore_color.rgb = rgb(d["fill"]); sp.line.fill.background(); sp.shadow.inherit = False
            sp.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            if d["s"]:
                set_runs(sp.text_frame, d["s"], d["size"], d["color"], d["bold"], d["bold_first"])
        elif kind == "text":
            n = d["s"].count("\n") + 1
            w, h = 6.8 if len(d["s"]) > 30 else 1.6, 0.14 * n
            x = d["x"] - (w if d["ha"] == "right" else w / 2 if d["ha"] == "center" else 0)
            y = d["y"] - (h if d["va"] == "bottom" else h / 2 if d["va"] == "center" else 0)
            tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tb.text_frame.vertical_anchor = {"top": MSO_ANCHOR.TOP, "center": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[d["va"]]
            al = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[d["ha"]]
            set_runs(tb.text_frame, d["s"], d["size"], d["color"], d["bold"], align=al)
        elif kind == "arrow":
            c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(d["x1"]), Inches(d["y1"]), Inches(d["x2"]), Inches(d["y2"]))
            c.line.color.rgb = rgb(d["color"]); c.line.width = Pt(d["lw"])
            if d["dash"]:
                c.line.dash_style = MSO_LINE.DASH
            if d["head"]:
                ln = c.line._get_or_add_ln()
                etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
        elif kind == "panel":
            W, H = d["w"], d["h"]
            fig = plt.figure(figsize=(W, H))
            ax = add_axes(fig, W, H, d["l"], d["t"], W - d["l"] - d["r"], H - d["t"] - d["b"])
            d["fn"](ax)
            path = os.path.join(PANELS, f"{d['key']}.png")
            fig.savefig(path, dpi=DPI)
            plt.close(fig)
            slide.shapes.add_picture(path, Inches(d["x"]), Inches(d["y"]), Inches(W), Inches(H))
    # letters stay on top of the panel pictures
    for sp in list(slide.shapes):
        if sp.has_text_frame and sp.text_frame.text in list("abcdefg"):
            slide.shapes._spTree.remove(sp._element); slide.shapes._spTree.append(sp._element)


# ---------------------------------------------------------------- plotting helpers
def forest(ax, rows, xlabel, xlim=None, band=None, gap=1.0, zero=True, labelsize=6):
    """rows: (label, est, lo, hi, colour); est None makes a group heading."""
    y, ticks, labels, heads = 0, [], [], []
    for label, est, lo, hi, color in rows:
        ticks.append(y); labels.append(label)
        if est is None:
            heads.append(len(labels) - 1)
        else:
            ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot(est, y, "o", ms=3.2, mfc="white" if lo <= 0 <= hi else color, mec=color, mew=0.8, zorder=3)
        y += gap
    if band:
        ax.axvspan(-band, band, color=PALE, lw=0, zorder=0)
    if zero:
        ax.axvline(0, color=GREY, lw=0.6, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=labelsize)
    for i in heads:
        ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_ylim(y - gap + 0.6, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    s = Scene("Fig1", 5.42)
    L = 0.2
    s.letter("a", 0, 0)
    x = L
    for i, (t, m) in enumerate([("T0", "Month 0"), ("T1", "Month 2"), ("T2", "Month 4"), ("T3", "Month 6")]):
        s.box(x, 0.2, 0.52, 0.4, f"{t}\n{m}", fill=BLUE, color="white", bold_first=True, size=6.5)
        s.box(x + 0.55, 0.2, 0.86, 0.4, "14-day burst\nEMA, ring, phone", fill=T_AUTO, size=6)
        if i < 3:
            s.arrow(x + 1.43, 0.4, x + 1.80, 0.4)
        x += 1.823
    s.text(L, 0.65, "Core-sample visit: 5 min adaptation → 5 min resting ECG → 8 min resting EEG (eyes open and closed, random order) → 5 min serial subtraction →\n5 min recovery → questionnaires → blinded clinician interview", size=5.8, color=GREY)

    s.letter("b", 0, 1.03)
    cols = [(L, 1.55), (L + 1.58, 2.62), (L + 4.23, 1.31), (L + 5.57, 1.31)]
    head = ["Measure", "Instrument and setting", "Core (n = 300)", "Extended (n = 300)"]
    rows = [
        ("Resting ECG, lnRMSSD", "BIOPAC with ECG100C, lead II, 1,000 Hz", "All waves", "T0, T3"),
        ("Resting EEG, posterior alpha", "actiCHamp Plus, 32 channels, 1,000 Hz", "All waves", "–"),
        ("Nocturnal RMSSD", "Oura Ring 4; Polar H10 in 40 core participants", "All waves", "All waves"),
        ("EMA, passive sensing", "mindLAMP, 3 prompts per day", "All waves", "All waves"),
        ("Self-report", "Stress scales, PHQ-9, GAD-7, DASS-21", "All waves", "All waves"),
        ("Clinician rating", "CGI-S, C-SSRS, blind to devices", "All waves", "All waves"),
    ]
    y = 1.23
    for (cx, cw), t in zip(cols, head):
        s.box(cx, y, cw, 0.2, t, fill="#E4E4E4", bold=True, size=6.2, round_=False)
    y += 0.215
    for row in rows:
        for (cx, cw), t in zip(cols, row):
            s.box(cx, y, cw, 0.185, t, fill="#F6F6F6", size=6.2, round_=False)
        y += 0.2

    s.letter("c", 0, 2.72)
    stages = [("Screened", "n = 2,000", PALE, ""), ("Eligible", "n = 1,807", PALE, "193 ineligible"),
              ("Consented", "n = 1,132", PALE, "675 not consented"), ("Enrolled, T0", "n = 600", T_AUTO, "532 not sampled"),
              ("T1", "n = 553 (92.2%)", T_AUTO, ""), ("T2", "n = 521 (86.8%)", T_AUTO, ""), ("T3", "n = 498 (83.0%)", T_AUTO, "")]
    bw, gap = 0.82, 0.19
    x = L
    for i, (a, b, f, note) in enumerate(stages):
        s.box(x, 2.92, bw, 0.36, f"{a}\n{b}", fill=f, bold_first=True, size=6)
        if i:
            s.arrow(x - gap + 0.01, 3.1, x - 0.01, 3.1)
        if note:
            s.text(x - gap / 2, 3.31, note, size=5.5, ha="center", color=GREY)
        x += bw + gap

    s.letter("d", 0, 3.58)
    yy, hh = 4.3, 0.56
    s.box(L, yy, 1.1, hh, "Educational stress\nacademic, employment,\ninterpersonal (t)", fill=T_STRESS, bold_first=True, size=6)
    s.box(L + 1.72, yy, 0.98, hh, "Autonomic state\nlnRMSSD (t + 1)", fill=T_AUTO, bold_first=True, size=6)
    s.box(L + 3.32, yy, 0.98, hh, "Subclinical distress\ncomposite (t + 2)", fill=T_CRISIS, bold_first=True, size=6)
    s.arrow(L + 1.12, yy + hh / 2, L + 1.70, yy + hh / 2, color=INK, lw=1.3)
    s.arrow(L + 2.72, yy + hh / 2, L + 3.30, yy + hh / 2, color=INK, lw=1.3)
    s.text(L + 1.41, yy + hh / 2 + 0.05, "H1", size=6.5, ha="center", bold=True)
    s.text(L + 3.01, yy + hh / 2 + 0.05, "H2", size=6.5, ha="center", bold=True)
    s.box(L + 0.86, 3.72, 1.1, 0.26, "Appraisal, mindset (H4a)", fill=PALE, size=6)
    s.box(L + 2.46, 3.72, 1.1, 0.26, "Emotion regulation (H4b)", fill=PALE, size=6)
    s.arrow(L + 1.41, 3.99, L + 1.41, yy + hh / 2 - 0.02, color=GREY)
    s.arrow(L + 3.01, 3.99, L + 3.01, yy + hh / 2 - 0.02, color=GREY)
    s.arrow(L + 3.81, yy + hh + 0.02, L + 3.81, yy + hh + 0.18, color=GREY, dash=True, head=False)
    s.arrow(L + 3.81, yy + hh + 0.18, L + 0.55, yy + hh + 0.18, color=GREY, dash=True, head=False)
    s.arrow(L + 0.55, yy + hh + 0.18, L + 0.55, yy + hh + 0.02, color=GREY, dash=True)
    s.text(L + 2.18, yy + hh + 0.21, "Reverse paths estimated in the same model; indirect effect a × b is H3", size=5.6, ha="center", color=GREY)

    rows_q = sheet("Fig 1e")
    q = [("Laboratory ECG sessions", rows_q[0]["%"]), ("EEG sessions", rows_q[1]["%"]),
         ("Ring bursts, ≥5 valid nights", rows_q[3]["%"]), ("Ring nights", rows_q[2]["%"]),
         ("Step person-days", rows_q[5]["%"]), ("EMA prompts answered", rows_q[6]["%"])]

    def e(ax):
        yp = np.arange(len(q))
        ax.barh(yp, [v for _, v in q], height=0.62, color=BLUE)
        for i, (_, v) in enumerate(q):
            ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=6)
        ax.set_yticks(yp); ax.set_yticklabels([k for k, _ in q]); ax.invert_yaxis()
        ax.set_xlim(0, 115); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Valid or answered (%)")
        ax.tick_params(axis="y", length=0)
    s.letter("e", 4.72, 3.58)
    s.panel("fig1e", 4.72, 3.58, 2.367, 1.72, 1.12, 0.2, 0.05, 0.36, e)
    return s


# ---------------------------------------------------------------- Fig. 2
def fig2():
    s = Scene("Fig2", 4.5)
    paths = {r["Path"]: r for r in sheet("Fig 2a,b")}
    code = {"Stress": "S", "lnRMSSD": "H", "Distress": "D"}

    def beta(a, b):
        return paths[f"{a} to {b}"]["Standardized β"]

    s.letter("a", 0, 0)
    xl, xr, nw, nh = 0.62, 2.62, 0.62, 0.34
    ys = {"Stress": 0.3, "lnRMSSD": 0.95, "Distress": 1.6}
    tints = {"Stress": T_STRESS, "lnRMSSD": T_AUTO, "Distress": T_CRISIS}
    for k, y in ys.items():
        s.text(xl - 0.06, y + nh / 2, k, size=6, ha="right", va="center", color=GREY)
        s.box(xl, y, nw, nh, f"{code[k]} (t)", fill=tints[k], bold=True, size=6.5)
        s.box(xr, y, nw, nh, f"{code[k]} (t + 1)", fill=tints[k], bold=True, size=6.5)
    x1, x2 = xl + nw + 0.02, xr - 0.02

    labels = []

    def line(a, b, frac, color=GREY, lw=0.8, dash=False, label_color=GREY, off=0.0, bold=False):
        y1, y2 = ys[a] + nh / 2 + off, ys[b] + nh / 2 + off
        s.arrow(x1, y1, x2, y2, color=color, lw=lw, dash=dash)
        xt, yt = x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac
        labels.append((xt, yt, num(beta(a, b)), label_color, bold))

    for k in ys:
        line(k, k, 0.2)
    line("Stress", "Distress", 0.33, off=0.04)
    line("Distress", "Stress", 0.82, dash=True, off=-0.04)
    line("Distress", "lnRMSSD", 0.78, dash=True, off=-0.08)
    line("lnRMSSD", "Stress", 0.5, dash=True, off=-0.08)
    line("Stress", "lnRMSSD", 0.2, color=BLUE, lw=1.5, label_color=BLUE, off=0.07, bold=True)
    line("lnRMSSD", "Distress", 0.2, color=BLUE, lw=1.5, label_color=BLUE, off=0.07, bold=True)
    for xt, yt, v, c_, bold_ in labels:
        s.box(xt - 0.13, yt - 0.05, 0.26, 0.1, v, fill="white", color=c_, size=5.8, bold=bold_, round_=False)

    def b(ax):
        order = ["Stress to lnRMSSD", "lnRMSSD to Distress", "Stress to Distress", "Distress to Stress", "Distress to lnRMSSD",
                 "lnRMSSD to Stress", "Stress to Stress", "lnRMSSD to lnRMSSD", "Distress to Distress"]
        rows = []
        for o in order:
            r = paths[o]
            a_, b_ = o.split(" to ")
            lo, hi = ci(r["95% CI"])
            rows.append((f"{code[a_]} → {code[b_]}", r["Standardized β"], lo, hi, BLUE if r["Role"] == "Focal" else GREY))
        forest(ax, rows, "Standardized within-person lag (95% CI)", band=0.07, xlim=(-0.27, 0.47))
    s.letter("b", 3.55, 0)
    s.panel("fig2b", 3.55, 0, 3.537, 2.2, 0.52, 0.08, 0.1, 0.36, b)

    def c(ax):
        rows = sheet("Fig 2c")
        order = ["Stress to lnRMSSD", "lnRMSSD to Distress", "Stress to Distress", "Distress to Stress", "Distress to lnRMSSD", "lnRMSSD to Stress"]
        for i, o in enumerate(order):
            for m, color, dy in [("Traditional cross-lagged model", MID, -0.17), ("RI-CLPM, main", BLUE, 0.17)]:
                r = [x for x in rows if x["Model"] == m and x["Path"] == o][0]
                lo, hi = ci(r["Standardized 95% CI"]); v = r["Standardized β"]
                ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=1.0, solid_capstyle="butt")
                ax.plot(v, i + dy, "o", ms=3.2, mec=color, mew=0.8, mfc="white" if lo <= 0 <= hi else color)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_yticks(range(6))
        ax.set_yticklabels([" → ".join(code[k] for k in o.split(" to ")) for o in order])
        ax.set_ylim(5.6, -0.6); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_xlabel("Standardized lag (95% CI)")
        ax.plot([], [], "o-", color=MID, ms=3, lw=1, label="Traditional CLPM")
        ax.plot([], [], "o-", color=BLUE, ms=3, lw=1, label="RI-CLPM")
        ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.3)
    s.letter("c", 0, 2.32)
    s.panel("fig2c", 0, 2.32, 3.45, 2.18, 0.52, 0.08, 0.1, 0.36, c)

    def d(ax):
        rows = {r["Pair"]: r for r in sheet("Fig 2d")}
        order = ["Stress – lnRMSSD", "lnRMSSD – Distress", "Stress – Distress"]
        xs = np.arange(3); w = 0.36
        ax.bar(xs - w / 2, [rows[o]["Between persons, r"] for o in order], w * 0.95, color=ORANGE, label="Between persons")
        ax.bar(xs + w / 2, [rows[o]["Within persons, r"] for o in order], w * 0.95, color=BLUE, label="Within persons")
        for i, o in enumerate(order):
            for dx, k in [(-w / 2, "Between persons, r"), (w / 2, "Within persons, r")]:
                v = rows[o][k]
                ax.text(i + dx, v + (0.012 if v > 0 else -0.012), num(v), ha="center",
                        va="bottom" if v > 0 else "top", fontsize=5.5)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(["S – H", "H – D", "S – D"]); ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Pearson r"); ax.set_ylim(-0.24, 0.4); ax.spines["bottom"].set_visible(False)
        ax.legend(loc="upper left", handlelength=1, borderaxespad=0.1, labelspacing=0.3)
    s.letter("d", 3.55, 2.32)
    s.panel("fig2d", 3.55, 2.32, 3.537, 2.18, 0.52, 0.08, 0.1, 0.24, d)
    return s


# ---------------------------------------------------------------- Fig. 3
def fig3():
    s = Scene("Fig3", 4.5)
    R = {r["Specification"]: r for r in sheet("Fig 3a")}

    def row(label, key, color=GREY):
        lo, hi = ci(R[key]["95% CI"])
        return (label, R[key]["Indirect effect"], lo, hi, color)

    rows = [
        ("Main model", None, None, None, None),
        row("Laboratory lnRMSSD, composite distress", "Main model: laboratory lnRMSSD, composite distress", BLUE),
        ("Outcome", None, None, None, None),
        row("Self-report only", "Distress: self-report only (PHQ-9, GAD-7)"),
        row("Clinician only", "Distress: clinician only (CGI-S, C-SSRS)"),
        row("Latent factor", "Distress: latent factor over four waves"),
        row("DASS-21 stress", "Distress: DASS-21 stress"),
        ("Mediator or predictor", None, None, None, None),
        row("High-frequency power", "Mediator: high-frequency power"),
        row("Nocturnal ring lnRMSSD", "Mediator: nocturnal ring lnRMSSD"),
        row("Ring, laboratory visits only", "Mediator: ring lnRMSSD kept only at visits with laboratory ECG"),
        row("Objective load from platform logs", "Stress: objective load from platform logs"),
        ("Adjustment and centring", None, None, None, None),
        row("No auxiliaries", "No auxiliaries"),
        row("Sex, trauma on random intercepts", "Sex and childhood trauma on the random intercepts"),
        row("Sleep, examinations, treatment out", "Within-person sleep, examination proximity and treatment regressed out"),
        row("After-treatment assessments removed", "Assessments after treatment began removed"),
        row("Cohort-by-wave centred", "Centred on cohort-by-wave means"),
        row("School-by-wave centred", "Centred on school-by-wave means"),
        ("Concurrent effects", None, None, None, None),
        row("lnRMSSD → distress", "Concurrent lnRMSSD to distress in place of residual covariance"),
        row("Stress → distress", "Concurrent stress to distress in place of residual covariance"),
        row("Stress → lnRMSSD", "Concurrent stress to lnRMSSD added"),
        ("Missing distress scores imputed", None, None, None, None),
        row("No shift", "Multiple imputation of distress, no shift"),
        row("Imputed scores +0.3 SD", "Multiple imputation of distress, imputed scores raised by 0.3 SD"),
        row("Imputed scores +0.6 SD", "Multiple imputation of distress, imputed scores raised by 0.6 SD"),
        ("Core sample (n = 300)", None, None, None, None),
        row("lnRMSSD", "Core sample only", ORANGE),
        row("Posterior alpha", "Mediator: posterior alpha (core sample)", ORANGE),
        row("lnRMSSD and alpha index", "Mediator: lnRMSSD and alpha index (core sample)", ORANGE),
        row("Frontal alpha asymmetry", "Mediator: frontal alpha asymmetry (core sample)", ORANGE),
        row("Recovery after the stressor", "Mediator: recovery after the stressor (core sample)", ORANGE),
    ]

    def a(ax):
        forest(ax, rows, "Unstandardized indirect effect a × b (95% Monte Carlo CI)", xlim=(-0.03, 0.135), labelsize=5.8)
        ax.axvline(0.025, color=BLUE, lw=0.5, ls=(0, (2, 2)), zorder=0)
    s.letter("a", 0, 0)
    s.panel("fig3a", 0, 0, 4.45, 4.5, 1.72, 0.06, 0.08, 0.32, a)

    slopes = sheet("Fig 3b,c")

    def simple(path, xlabel, color):
        def fn(ax):
            rr = [r for r in slopes if r["Path"] == path]
            xs = [r["Moderator level (SD)"] for r in rr]
            ys = [r["Simple slope"] for r in rr]
            jn = rr[0]["Slope not significant from (SD)"]
            ax.axvspan(jn, 2, color=PALE, lw=0, zorder=0)
            ax.plot(xs, ys, color=color, lw=1.0, zorder=2)
            for r, x, y in zip(rr, xs, ys):
                lo, hi = ci(r["95% CI"])
                ax.plot([x, x], [lo, hi], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
                ax.plot(x, y, "o", ms=3.4, mfc="white" if lo <= 0 <= hi else color, mec=color, mew=0.8, zorder=3)
            ax.axhline(0, color=GREY, lw=0.6)
            ax.set_xlim(-1.5, 2); ax.set_xticks([-1, 0, 1, 2]); ax.set_ylim(-0.52, 0.14)
            ax.set_xlabel(xlabel); ax.set_ylabel("Simple slope (95% CI)")
            ax.text(jn + 0.06, -0.49, f"n.s. above {jn:.2f} SD", fontsize=5.5, color=GREY, va="bottom")
        return fn
    s.letter("b", 4.55, 0)
    s.panel("fig3b", 4.55, 0, 2.537, 2.2, 0.5, 0.08, 0.08, 0.36, simple("Stress to lnRMSSD", "Challenge appraisal and mindset (SD)", BLUE))
    s.letter("c", 4.55, 2.28)
    s.panel("fig3c", 4.55, 2.28, 2.537, 2.2, 0.5, 0.08, 0.08, 0.36, simple("lnRMSSD to distress", "Emotion regulation (SD)", GREY))
    return s


# ---------------------------------------------------------------- Fig. 4
def fig4():
    s = Scene("Fig4", 4.25)
    ml = {r["Path"]: float(r["b"]) for r in read_csv(MULTILEVEL)}
    ag = read_csv(AGREEMENT)

    def binned(name, slope, color, xlabel, ylabel):
        rows = sheet(name)

        def fn(ax):
            x = np.array([r["Predictor mean"] for r in rows]); y = np.array([r["Outcome mean"] for r in rows])
            n = np.array([r["n"] for r in rows], float)
            xm, ym = np.average(x, weights=n), np.average(y, weights=n)
            ends = np.array([x.min(), x.max()])
            ax.axhline(0, color=GREY, lw=0.5)
            ax.plot(ends, ym + slope * (ends - xm), color=color, lw=1.0)
            for r, xi, yi in zip(rows, x, y):
                lo, hi = ci(r["95% CI"])
                ax.plot([xi, xi], [lo, hi], color=color, lw=0.9, solid_capstyle="butt")
            ax.plot(x, y, "o", ms=3.2, color=color, mec="white", mew=0.4)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return fn

    def ba(name, row, xlabel):
        rows = sheet(name)
        keys = list(rows[0].keys())
        rec = ag[row]
        bias = float(rec["Bias"]); lo, hi = ci(rec["Limits of agreement"])

        def fn(ax):
            x = [r[keys[0]] for r in rows]; y = [r[keys[1]] for r in rows]
            many = len(x) > 500
            ax.scatter(x, y, s=2.2 if many else 4, color=BLUE, alpha=0.35 if many else 0.55, lw=0, rasterized=many)
            ax.axhline(bias, color=INK, lw=0.8)
            ax.axhline(lo, color=INK, lw=0.6, ls=(0, (3, 2))); ax.axhline(hi, color=INK, lw=0.6, ls=(0, (3, 2)))
            xr = ax.get_xlim()[1]
            for v in (lo, bias, hi):
                ax.text(xr, v, " " + num(v), va="center", ha="left", fontsize=5.5)
            ax.set_xlim(right=xr)
            ax.set_xlabel(xlabel); ax.set_ylabel("Difference (ln ms)")
        return fn

    s.letter("a", 0, 0)
    s.panel("fig4a", 0, 0, 2.05, 2.08, 0.5, 0.08, 0.08, 0.42, binned("Fig 4a", ml["Daily stress to that night's lnRMSSD"], BLUE, "Daily stress\n(within person)", "Nocturnal lnRMSSD\n(within person)"))
    s.letter("b", 2.1, 0)
    s.panel("fig4b", 2.1, 0, 2.05, 2.08, 0.5, 0.08, 0.08, 0.42, binned("Fig 4b", ml["Nocturnal lnRMSSD to next-day negative affect"], ORANGE, "Nocturnal lnRMSSD\n(within person)", "Next-day negative affect\n(within person)"))
    s.letter("c", 0, 2.17)
    s.panel("fig4c", 0, 2.17, 2.05, 2.08, 0.5, 0.08, 0.26, 0.32, ba("Fig 4c", 0, "Mean of ring and H10 (ln ms)"))
    s.letter("d", 2.1, 2.17)
    s.panel("fig4d", 2.1, 2.17, 2.05, 2.08, 0.5, 0.08, 0.26, 0.32, ba("Fig 4d", 1, "Mean of ring and laboratory (ln ms)"))

    D = {r["Parameter"]: r for r in sheet("Fig 4e")}

    def pr(label, key, color):
        lo, hi = ci(D[key]["95% CrI"])
        return (label, D[key]["Standardized median"], lo, hi, color)
    rows = [
        pr("Stress → same-night lnRMSSD", "Daily stress to that night's lnRMSSD", BLUE),
        pr("lnRMSSD → next-day affect", "Nocturnal lnRMSSD to next-day negative affect", BLUE),
        pr("Affect → same-night lnRMSSD", "Daily negative affect to that night's lnRMSSD", GREY),
        pr("lnRMSSD → next-day stress", "Nocturnal lnRMSSD (day t − 1) to Stress (day t)", GREY),
        pr("Stress → next-day affect", "Stress (day t − 1) to Negative affect (day t)", GREY),
        pr("Affect → next-day stress", "Negative affect (day t − 1) to Stress (day t)", GREY),
        pr("Stress → next-night lnRMSSD", "Stress (day t − 1) to Nocturnal lnRMSSD (day t)", GREY),
        pr("Affect → next-night lnRMSSD", "Negative affect (day t − 1) to Nocturnal lnRMSSD (day t)", GREY),
        pr("Stress carry-over", "Stress (day t − 1) to Stress (day t)", LIGHT),
        pr("Affect carry-over", "Negative affect (day t − 1) to Negative affect (day t)", LIGHT),
        pr("lnRMSSD carry-over", "Nocturnal lnRMSSD (day t − 1) to Nocturnal lnRMSSD (day t)", LIGHT),
    ]

    def e(ax):
        forest(ax, rows, "Standardized within-person path\n(posterior median, 95% CrI)", xlim=(-0.15, 0.37))
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    s.letter("e", 4.25, 0)
    s.panel("fig4e", 4.25, 0, 2.837, 4.25, 1.3, 0.08, 0.08, 0.42, e)
    return s


# ---------------------------------------------------------------- Fig. 5
def fig5():
    s = Scene("Fig5", 2.05)

    def bic(name, kcol, gcol, groups, xlabel, ylim):
        rows = sheet(name)

        def fn(ax):
            for g, label, color in groups:
                rr = [r for r in rows if r[gcol] == g and r[kcol] >= 2]
                ax.plot([r[kcol] for r in rr], [r["BIC minus minimum BIC"] for r in rr], "o-", color=color, ms=3, mec="white", mew=0.4, label=label)
            ax.set_xticks(range(2, 7)); ax.set_xlabel(xlabel); ax.set_ylabel("BIC − minimum BIC"); ax.set_ylim(*ylim)
            ax.legend(loc="upper right", handlelength=1.6, labelspacing=0.3, borderaxespad=0)
        return fn

    def b(ax):
        rows = sheet("Fig 5b")
        waves = ["T0", "T1", "T2", "T3"]
        names = {"P1": "P1 stable low", "P2": "P2 improving", "P3": "P3 slow rise", "P4": "P4 elevated", "P5": "P5 escalating"}
        for p in ["P1", "P2", "P3", "P4", "P5"]:
            rr = [r for r in rows if r["Profile"] == p]
            y = [r["Distress, mean"] for r in rr]
            ax.plot(range(4), y, "-", color=PROFILE[p], lw=1.0)
            for i, r in enumerate(rr):
                lo, hi = ci(r["95% CI"])
                ax.plot([i, i], [lo, hi], color=PROFILE[p], lw=0.8, solid_capstyle="butt")
            ax.plot(range(4), y, "o", ms=2.8, color=PROFILE[p], mec="white", mew=0.3)
        for p, yy in [("P5", 1.57), ("P4", 0.62), ("P3", -0.13), ("P2", -0.52), ("P1", -0.83)]:
            ax.text(3.08, yy, names[p], color=PROFILE[p], fontsize=5.6, va="center")
        ax.set_xlim(-0.2, 4.35); ax.set_xticks(range(4)); ax.set_xticklabels(waves); ax.set_ylabel("Composite distress score (z)")
        ax.set_xlabel("Wave")

    def c(ax):
        rows = sheet("Fig 5c")[:5]
        M = np.array([[r[f"P{i} mean"] for i in range(1, 6)] for r in rows])
        cmap = LinearSegmentedColormap.from_list("d", ["#2C7FB8", "#F7F7F7", "#D9622B"])
        ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -0.8, 0.8), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, num(round(float(M[i, j]), 3)), ha="center", va="center", fontsize=5.6,
                        color="white" if abs(M[i, j]) > 0.62 else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"P{i}" for i in range(1, 6)])
        ax.set_yticks(range(5)); ax.set_yticklabels(["Appraisal, mindset", "Emotion regulation", "Stress, T0", "lnRMSSD, T0", "Childhood trauma"])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Profile")

    def e(ax):
        rows = sheet("Supp Fig 1b")
        M = np.array([[r["Probability"] for r in rows if r["From"] == f"S{i}"] for i in range(1, 6)])
        cmap = LinearSegmentedColormap.from_list("b", ["#F7FBFF", "#8FB8E0", "#1B5A96"])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=0.75, aspect="auto")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, num(round(float(M[i, j]), 3)), ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 0.5 else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"S{i}" for i in range(1, 6)])
        ax.set_yticks(range(5)); ax.set_yticklabels([f"S{i}" for i in range(1, 6)])
        ax.set_xlabel("State at next wave"); ax.set_ylabel("State at current wave"); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def f(ax):
        rows = sheet("Supp Fig 1c")
        bottom = np.zeros(4)
        for r, color in zip(rows, STATE):
            v = np.array([r[f"T{i}, %"] for i in range(4)])
            ax.bar(range(4), v, bottom=bottom, width=0.68, color=color, edgecolor="white", lw=0.4, label=r["State"])
            for i in range(4):
                if v[i] >= 7:
                    ax.text(i, bottom[i] + v[i] / 2, f"{v[i]:.1f}", ha="center", va="center", fontsize=5.3,
                            color=INK if color in STATE[:2] else "white")
            bottom += v
        ax.set_xticks(range(4)); ax.set_xticklabels(["T0", "T1", "T2", "T3"]); ax.set_ylabel("State prevalence (%)"); ax.set_ylim(0, 100.5)
        ax.set_xlabel("Wave")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=0.9, labelspacing=0.3, borderaxespad=0.2, reverse=True)

    W1, W2, W3 = 2.15, 2.5, 2.437
    s.letter("a", 0, 0)
    s.panel("fig5a", 0, 0, W1, 2.05, 0.46, 0.1, 0.08, 0.36, bic("Fig 5a", "Classes", "Indicators", [("Five indicators", "Five indicators", BLUE), ("Three symptom indicators", "Three symptom indicators", ORANGE)], "Profiles (k)", (-12, 250)))
    s.letter("b", W1, 0)
    s.panel("fig5b", W1, 0, W2, 2.05, 0.46, 0.1, 0.02, 0.36, b)
    s.letter("c", W1 + W2, 0)
    s.panel("fig5c", W1 + W2, 0, W3, 2.05, 0.92, 0.1, 0.06, 0.36, c)

    t = Scene("SupplementaryFig1", 2.05)
    t.letter("a", 0, 0)
    t.panel("figS1a", 0, 0, W1, 2.05, 0.46, 0.1, 0.08, 0.36, bic("Supp Fig 1a", "States", "Model", [("LTA without random intercept", "LTA", MID), ("Random-intercept LTA", "RI-LTA", BLUE)], "Latent states (k)", (-60, 2050)))
    t.letter("b", W1, 0)
    t.panel("figS1b", W1, 0, W2, 2.05, 0.46, 0.1, 0.1, 0.36, e)
    t.letter("c", W1 + W2, 0)
    t.panel("figS1c", W1 + W2, 0, W3, 2.05, 0.5, 0.1, 0.42, 0.36, f)
    return s, t


# ---------------------------------------------------------------- Fig. 6
def fig6():
    s = Scene("Fig6", 4.3)
    P = sheet("Fig 6a")

    def pr(pop, model, label, color):
        r = [x for x in P if x["Population"] == pop and x["Model"] == model][0]
        lo, hi = ci(r["95% CI"])
        return (label, r["AUC change from self-report"], lo, hi, color)
    rows = [("All transitions (n = 1,538)", None, None, None, None)]
    for m, l in [("Self-report + physiology", "+ physiology"), ("Self-report + passive sensing", "+ passive sensing"), ("Self-report + momentary assessment", "+ EMA"), ("All features", "All features")]:
        rows.append(pr("All transitions", m, l, BLUE))
    rows.append(("New onset (n = 1,005)", None, None, None, None))
    for m, l in [("Self-report + physiology", "+ physiology"), ("Self-report + passive sensing", "+ passive sensing"), ("Self-report + momentary assessment", "+ EMA"), ("All features", "All features")]:
        rows.append(pr("New onset", m, l, ORANGE))

    def a(ax):
        forest(ax, rows, "ΔAUC against self-report (95% bootstrap CI)", xlim=(-0.02, 0.052))
    s.letter("a", 0, 0)
    s.panel("fig6a", 0, 0, 4.5, 2.1, 1.3, 0.06, 0.1, 0.34, a)

    def b(ax):
        r = sheet("Fig 6b")
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)))
        for x in r:
            lo, hi = ci(x["95% CI"])
            ax.plot([x["Mean predicted risk"]] * 2, [lo, hi], color=BLUE, lw=0.9, solid_capstyle="butt")
        ax.plot([x["Mean predicted risk"] for x in r], [x["Observed proportion"] for x in r], "o", ms=3, color=BLUE, mec="white", mew=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
        ax.text(0.97, 0.05, "Slope 0.93\nIntercept 0.003", ha="right", va="bottom", fontsize=5.6)
    s.letter("b", 4.6, 0)
    s.panel("fig6b", 4.6, 0, 2.487, 2.1, 0.5, 0.06, 0.2, 0.34, b)

    def c(ax):
        r = [x for x in sheet("Fig 6c") if x["Population"] == "All transitions"]
        for m, label, color in [("Self-report", "Self-report", GREY), ("All features", "All features", BLUE)]:
            rr = [x for x in r if x["Model"] == m]
            ax.plot([x["Risk threshold"] for x in rr], [x["Net benefit"] for x in rr], "o-", color=color, ms=3, mec="white", mew=0.4, label=label)
        rr = [x for x in r if x["Model"] == "Self-report"]
        ax.plot([x["Risk threshold"] for x in rr], [x["Net benefit of contacting all"] for x in rr], color=LIGHT, lw=1.0, ls=(0, (3, 2)), label="Contact all")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_ylim(-0.05, 0.36); ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_xlabel("Risk threshold"); ax.set_ylabel("Net benefit")
        ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.3, borderaxespad=0)
    s.letter("c", 0, 2.2)
    s.panel("fig6c", 0, 2.2, 3.3, 2.1, 0.5, 0.08, 0.15, 0.34, c)

    def d(ax):
        r = sheet("Fig 6d")[:10]
        passive = {"Location entropy", "Night-time screen use", "Steps", "Time at home"}
        phys = {"Nocturnal ring lnRMSSD"}
        colors = [ORANGE if x["Feature"] in passive else BLUE if x["Feature"] in phys else LIGHT for x in r]
        y = np.arange(len(r))
        ax.barh(y, [x["Mean absolute Shapley value"] for x in r], height=0.68, color=colors)
        ax.set_yticks(y); ax.set_yticklabels([x["Feature"] for x in r]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Mean absolute Shapley value (log odds)")
        for label, color, yy in [("Self-report", LIGHT, 6.6), ("Passive sensing", ORANGE, 7.6), ("Physiology", BLUE, 8.6)]:
            ax.barh(yy, 0.018, height=0.5, left=0.29, color=color)
            ax.text(0.315, yy, label, va="center", fontsize=5.8)
    s.letter("d", 3.4, 2.2)
    s.panel("fig6d", 3.4, 2.2, 3.687, 2.1, 1.12, 0.08, 0.12, 0.34, d)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PANELS, exist_ok=True)
    s5, s_supp = fig5()
    scenes = [fig1(), fig2(), fig3(), fig4(), s5, fig6(), s_supp]
    prs = Presentation()
    prs.slide_width = Inches(WIDTH)
    prs.slide_height = Inches(max(s.height for s in scenes))
    for s in scenes:
        render_mpl(s)
        render_ppt(prs, s)
    prs.save(os.path.join(OUT, "Figures_editable.pptx"))


if __name__ == "__main__":
    main()
