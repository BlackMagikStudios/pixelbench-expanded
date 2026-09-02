"""Render a labelled montage of one image through every distortion category.

Useful for docs and for eyeballing what suite v1 does. Not part of scoring.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from .data import load_art, list_images
from .distort import make_spec, distort, category_names


def _tile(img: np.ndarray, size: int) -> Image.Image:
    """Nearest-scale an RGB array to fit a size x size tile, centered. The
    distortion is generated at native resolution; this only upscales it for
    viewing (never smooth-downscales, so the pixels stay crisp)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    im = Image.fromarray(img).resize((nw, nh), Image.NEAREST)
    tile = Image.new("RGB", (size, size), (26, 26, 30))
    tile.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return tile


def make_preview(image_path: str, out_path: str, tile: int = 200,
                 severity: float = 1.0) -> str:
    image_id = os.path.basename(image_path)
    art = load_art(image_path, np.random.default_rng(0))
    # Cap each category's upscale so the distorted image is at most `tile` px:
    # the distortion happens on the native art, then the result is only ever
    # nearest-UPSCALED for display, never downscaled. This keeps every tile's
    # pixels crisp and comparable instead of smooth-shrinking the big ones.
    max_scale = max(1.0, tile / max(art.shape[:2]))
    cats = ["(native)"] + category_names()
    cols = 6 if len(cats) > 16 else 4       # contact-sheet for the full suite
    rows = (len(cats) + cols - 1) // cols
    pad, label_h = 8, 20
    cw, ch = tile + pad, tile + pad + label_h
    canvas = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for i, cat in enumerate(cats):
        if cat == "(native)":
            img = art
        else:
            spec = make_spec(cat, image_id, severity)
            spec.sx = min(spec.sx, max_scale)
            spec.sy = min(spec.sy, max_scale)
            img = distort(art, spec, np.random.default_rng(spec.seed))["img"]
        x = pad + (i % cols) * cw
        y = pad + (i // cols) * ch
        canvas.paste(_tile(img, tile), (x, y))
        draw.text((x + 2, y + tile + 4), cat, fill=(210, 210, 215))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    return out_path


def preview_dataset(data_dir: str, out_dir: str, n: int = 1,
                    severity: float = 1.0, tile: int = 200) -> list[str]:
    paths = list_images(data_dir)[:n]
    outs = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        outs.append(make_preview(p, os.path.join(out_dir, f"{stem}_suite.png"),
                                 tile=tile, severity=severity))
    return outs
