"""Render a markdown report from a pixel-bench results JSON."""
from __future__ import annotations

import json

import numpy as np

# display order + labels for methods we know; unknown methods still render
KNOWN_LABELS = {
    "fixer": "Pixel Art Fixer", "snapper": "PixelSnapper",
    "pixeldetector": "PixelDetector", "unfake": "unfake.py", "naive": "Naive",
}
MEDIAN_FIELDS = {"rel_err", "delta_e", "psnr", "palette_de", "ncolor_err", "sec"}
PCT_FIELDS = {"exact", "within1", "within5pct", "pixel_match", "grid_align", "rel_err"}


def _label(m):
    return KNOWN_LABELS.get(m, m)


def _agg(rows, field):
    vals = [r[field] for r in rows if field in r and "error" not in r]
    if not vals:
        return None
    v = float(np.median(vals) if field in MEDIAN_FIELDS else np.mean(vals))
    return 100 * v if field in PCT_FIELDS else v


def _fmt(field, v):
    if v is None:
        return "-"
    if field in PCT_FIELDS:
        return f"{v:.1f}"
    if field in ("delta_e", "psnr", "palette_de"):
        return f"{v:.1f}"
    if field == "sec":
        return f"{v:.2f}"
    return f"{v:.2f}"


def _rows_for(samples, method, category=None):
    return [r["methods"][method] for r in samples
            if "methods" in r and method in r["methods"]
            and (category is None or r["category"] == category)
            and "error" not in r["methods"][method]]


def _table(samples, methods, fields, headers, category=None):
    out = ["| Method | " + " | ".join(headers) + " |",
           "|" + "---|" * (len(fields) + 1)]
    for m in methods:
        rows = _rows_for(samples, method=m, category=category)
        cells = [_fmt(f, _agg(rows, f)) for f in fields]
        out.append(f"| {_label(m)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def render(payload) -> str:
    samples = [r for r in payload["samples"] if "methods" in r]
    methods = payload["methods"]
    cats = payload["aggregate"]["categories"]
    L = []
    W = L.append

    W("# pixel-bench results")
    W("")
    W(f"Suite **{payload['suite']}** at severity **{payload['severity']}** on "
      f"**{payload['dataset']['n_images']}** source images "
      f"({payload['n_scored']} scored inputs across {len(cats)} distortion "
      f"categories).")
    W("")
    W("Signals (see `docs/METRICS.md`): **exact/±1** native-resolution hit "
      "rate, **rel err** median resolution error, **grid align** (do the "
      "detected grid lines sit on the true grid, from ground-truth U/V), "
      "**pixel match** (share of native pixels exactly right), **ΔE** mean "
      "CIELAB colour error. Higher is better except rel err and ΔE.")
    W("")

    W("## Overall")
    W("")
    W(_table(samples, methods,
             ["exact", "within1", "rel_err", "grid_align", "pixel_match", "delta_e"],
             ["Exact %", "±1 %", "Rel err %", "Grid align %", "Pixel match %", "ΔE"]))
    W("")

    # colour, conditioned on exact resolution (avoids the over-segmentation confound)
    W("### Colour on the exact-resolution subset")
    W("")
    W("Raw ΔE flatters methods that over-segment. Conditioned on getting the "
      "native size exactly right, colour fidelity is like-for-like:")
    W("")
    out = ["| Method | n (exact) | ΔE | Pixel match % |",
           "|---|---|---|---|"]
    for m in methods:
        ex = [r["methods"][m] for r in samples
              if m in r["methods"] and "error" not in r["methods"][m]
              and r["methods"][m].get("exact")]
        if not ex:
            out.append(f"| {_label(m)} | 0 | - | - |")
            continue
        de = np.median([x["delta_e"] for x in ex if "delta_e" in x])
        pm = 100 * np.mean([x["pixel_match"] for x in ex if "pixel_match" in x])
        out.append(f"| {_label(m)} | {len(ex)} | {de:.1f} | {pm:.1f} |")
    W("\n".join(out))
    W("")

    W("## By distortion category")
    W("")
    for c in cats:
        n = sum(1 for r in samples if r["category"] == c)
        W(f"### {c} ({n} images)")
        W("")
        W(_table(samples, methods,
                 ["exact", "within1", "rel_err", "grid_align", "delta_e"],
                 ["Exact %", "±1 %", "Rel err %", "Grid align %", "ΔE"],
                 category=c))
        W("")

    W("## Runtime")
    W("")
    W(_table(samples, methods, ["sec"], ["Mean s / image"]))
    W("")
    W(f"Wall time: {payload.get('wall_s', '?')}s. "
      f"pixel-bench {payload.get('pixelbench_version', '?')}, "
      f"seed {payload.get('seed', '?')}.")
    W("")
    return "\n".join(L) + "\n"


def render_file(json_path, out_path):
    payload = json.load(open(json_path))
    md = render(payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path
