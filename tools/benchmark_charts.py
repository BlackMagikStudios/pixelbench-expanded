"""Render the showcase benchmark charts from a pixel-bench results JSON.

Pillow-only (no matplotlib), in the same dark "Retro Diffusion" house style as
the technical figures in pixel-art-fixer/docs: near-black surface with a faint
grid and a corner lime glow, Segoe UI type, lime = the Pixel Art Fixer, muted
steel = the other tools. Drawn at 2x and downsampled for crisp anti-aliasing.

    python tools/benchmark_charts.py results/example-run.json --out docs/images
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---- house palette (matches docs/HOW_IT_WORKS figures) ----
BG = (10, 10, 12)
ACCENT = (196, 241, 40)
TEXT = (236, 236, 229)
BODY = (206, 206, 198)
MUTED = (150, 150, 142)
LINE = (54, 54, 58)
GRID = (28, 28, 31)
TRACK = (26, 27, 32)
STEEL = (92, 101, 116)
S = 2  # supersample factor

ORDER = ["fixer", "pixeldetector", "unfake", "snapper", "naive"]
LABELS = {"fixer": "Pixel Art Fixer", "pixeldetector": "PixelDetector",
          "unfake": "unfake.py", "snapper": "PixelSnapper", "naive": "Naive"}

_FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts",
              "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]


def _font(names, size):
    for d in _FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def FB(size):  # bold display
    return _font(["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], size * S)


def FSB(size):  # semibold body
    return _font(["seguisb.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"], size * S)


def _text(d, xy, s, font, fill, anchor="la"):
    d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor)


def _rect(d, box, fill):
    d.rectangle([c * S for c in box], fill=fill)


def glow(w, h):
    """Faint 40px grid plus an upper-right radial lime glow, at device scale."""
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    step = 40 * S
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(16, 16, 18))
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(16, 16, 18))
    g = Image.new("L", (w, h), 0)
    ImageDraw.Draw(g).ellipse(
        [w - int(w * 0.5), -h, w + int(w * 0.2), int(h * 0.8)], fill=52)
    g = g.filter(ImageFilter.GaussianBlur(130 * S))
    return Image.composite(
        Image.blend(img, Image.new("RGB", (w, h), ACCENT), 0.11), img, g)


# ------------------------------------------------------------ data


def load_stats(path):
    d = json.load(open(path))
    s = [r for r in d["samples"] if "methods" in r]
    cats = d["aggregate"]["categories"]

    def rows(m, cat=None):
        return [r["methods"][m] for r in s if m in r["methods"]
                and "error" not in r["methods"][m]
                and (cat is None or r["category"] == cat)]

    overall = {m: {k: 100 * float(np.mean([x[k] for x in rows(m)]))
                   for k in ("exact", "within1", "grid_align")} for m in ORDER}
    heat = {m: [100 * float(np.mean([x["exact"] for x in rows(m, c)]))
                for c in cats] for m in ORDER}
    meta = {"n_inputs": d.get("n_scored"), "n_images": d["dataset"]["n_images"],
            "suite": d.get("suite"), "cats": cats}
    return overall, heat, meta


# ------------------------------------------------------------ chart 1: accuracy


def chart_accuracy(overall, meta, out):
    W, H = 1040, 384
    img = glow(W * S, H * S)
    d = ImageDraw.Draw(img)
    f_title = FB(22)
    f_sub = FSB(13)
    f_panel = FB(15)
    f_val = FB(14)
    f_name = FSB(13)

    _text(d, (30, 22), "HOW WELL EACH TOOL REBUILDS THE PIXEL ART", f_title, TEXT)
    _text(d, (30, 54),
          f"Pixel Art Fixer vs the field  ·  {meta['n_inputs']} distorted "
          f"images  ·  suite {meta['suite']}", f_sub, MUTED)

    panels = [("Exact native size", "exact"),
              ("Within 1 pixel", "within1"),
              ("Grid alignment", "grid_align")]
    left = 150
    top = 110
    bottom = H - 30
    gap = 34
    plot_w = (W - left - 30 - gap * 2) / 3
    plot_h = bottom - top
    row_h = plot_h / len(ORDER)
    bar_h = row_h * 0.5

    # method names (shared, far left), aligned to rows
    for i, m in enumerate(ORDER):
        cy = top + row_h * (i + 0.5)
        _text(d, (left - 16, cy), LABELS[m], f_name,
              TEXT if m == "fixer" else MUTED, anchor="rm")

    for pi, (ptitle, key) in enumerate(panels):
        px0 = left + pi * (plot_w + gap)
        _text(d, (px0, top - 28), ptitle, f_panel, TEXT)
        # 0 / 50 / 100% gridlines
        for gx in (0.0, 0.5, 1.0):
            x = px0 + gx * plot_w
            d.line([(x * S, (top - 8) * S), (x * S, bottom * S)],
                   fill=GRID, width=S)
        for i, m in enumerate(ORDER):
            cy = top + row_h * (i + 0.5)
            val = overall[m][key]
            _rect(d, (px0, cy - bar_h / 2, px0 + plot_w, cy + bar_h / 2), TRACK)
            bw = max(val / 100 * plot_w, 1.5)
            _rect(d, (px0, cy - bar_h / 2, px0 + bw, cy + bar_h / 2),
                  ACCENT if m == "fixer" else STEEL)
            label = f"{val:.0f}%"
            inside = bw > 44
            _text(d, (px0 + bw + (-8 if inside else 6), cy), label, f_val,
                  BG if (inside and m == "fixer") else (TEXT if inside else BODY),
                  anchor="rm" if inside else "lm")

    img = img.resize((W, H), Image.LANCZOS)
    img.save(out)
    return out


# ------------------------------------------------------------ chart 2: categories


def _ramp(t):
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (18, 19, 22)), (0.5, (92, 110, 26)), (1.0, ACCENT)]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
    return stops[-1][1]


def chart_categories(heat, meta, out):
    # categories as rows (scales to the full 43-category suite), methods as
    # columns; the fixer column is first so it lights up down the left edge.
    cats = meta["cats"]
    left, top = 172, 118
    cell_w, cell_h = 106, 22
    gapc = 3
    W = left + len(ORDER) * (cell_w + gapc) + 24
    H = top + len(cats) * (cell_h + gapc) + 24
    img = glow(W * S, H * S)
    d = ImageDraw.Draw(img)
    f_title = FB(22)
    f_sub = FSB(13)
    f_cell = FB(12)
    f_head = FSB(11)
    f_cat = FSB(12)

    _text(d, (30, 22), "EXACT PIXEL ART RECOVERED, BY DISTORTION TYPE",
          f_title, TEXT)
    _text(d, (30, 54),
          f"Share of images rebuilt at the exact native resolution across all "
          f"{len(cats)} distortion types  ·  brighter is better", f_sub, MUTED)

    # method column headers
    for i, m in enumerate(ORDER):
        cx = left + i * (cell_w + gapc) + cell_w / 2
        _text(d, (cx, top - 14), LABELS[m], f_head,
              TEXT if m == "fixer" else MUTED, anchor="mm")

    for j, c in enumerate(cats):
        cy = top + j * (cell_h + gapc)
        _text(d, (left - 12, cy + cell_h / 2), c, f_cat, BODY, anchor="rm")
        for i, m in enumerate(ORDER):
            cx = left + i * (cell_w + gapc)
            val = heat[m][j]
            _rect(d, (cx, cy, cx + cell_w, cy + cell_h), _ramp(val / 100))
            tcol = (14, 16, 10) if val >= 45 else BODY
            _text(d, (cx + cell_w / 2, cy + cell_h / 2), f"{val:.0f}",
                  f_cell, tcol, anchor="mm")

    img = img.resize((W, H), Image.LANCZOS)
    img.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--only", choices=["accuracy", "categories"],
                    help="render just one chart instead of both")
    args = ap.parse_args()
    overall, heat, meta = load_stats(args.results)
    os.makedirs(args.out, exist_ok=True)
    if args.only in (None, "accuracy"):
        print("wrote", chart_accuracy(
            overall, meta, os.path.join(args.out, "benchmark-accuracy.png")))
    if args.only in (None, "categories"):
        print("wrote", chart_categories(
            heat, meta, os.path.join(args.out, "benchmark-categories.png")))


if __name__ == "__main__":
    main()
