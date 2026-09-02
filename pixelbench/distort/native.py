"""Native-art damages: the cell-level "broken pixel art" chain.

These operate on the NATIVE (1x) art *before* it is upscaled, modelling damage
that is baked into the art itself rather than introduced by the upscale: dead
pixels, broken outlines, native anti-aliasing, colour crush, composite halos.
In pixel-bench the damaged native becomes the ground truth (the input is that
damaged native, upscaled), so a method is scored on faithfully recovering the
art it was actually given.

Ported from the reference degradation tooling; pure numpy / scipy / PIL.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .engine import _gauss


def activity_map(art: np.ndarray, thresh: int = 12) -> np.ndarray:
    """Per-native-pixel bool: differs from any 4-neighbour."""
    a = art.astype(np.int16)
    d = np.zeros(art.shape[:2], bool)
    dr = (np.abs(a[:, 1:] - a[:, :-1]) > thresh).any(2)
    d[:, 1:] |= dr
    d[:, :-1] |= dr
    dd = (np.abs(a[1:] - a[:-1]) > thresh).any(2)
    d[1:] |= dd
    d[:-1] |= dd
    return d


def native_aa(art: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Anti-alias the native art itself: blend a small blur into the crisp
    art, either everywhere (pixelization-tool look) or only on edge pixels
    (hand-AA look)."""
    act = activity_map(art)
    a = float(rng.uniform(0.35, 0.85))
    soft = _gauss(art.astype(np.float32), float(rng.uniform(0.5, 0.9)))
    if rng.random() < 0.5:
        m = _gauss(act.astype(np.float32), 0.6)[..., None]
        out = art + (soft - art) * np.clip(m, 0, 1) * a
    else:
        out = art + (soft - art) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def overquantize(art: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild colour crush relative to the art's own palette (never dithered)."""
    flat = art.reshape(-1, 3).astype(np.int64)
    n = len(np.unique(flat[:, 0] << 16 | flat[:, 1] << 8 | flat[:, 2]))
    k = int(n * float(rng.uniform(0.60, 0.85)))
    if n <= 4 or k < 2 or k >= n:
        return art
    q = Image.fromarray(art).quantize(colors=min(k, 256),
                                      dither=Image.Dither.NONE)
    return np.asarray(q.convert("RGB"))


def alpha_halo(art: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bake a bad-composite fringe: pixels bordering the background blend
    toward a halo colour (white / black / background)."""
    from scipy.ndimage import binary_dilation
    border = np.concatenate([art[0], art[-1], art[:, 0], art[:, -1]])
    vals, cnt = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    bg = vals[cnt.argmax()]
    is_bg = (np.abs(art.astype(np.int16) - bg).max(2) < 12)
    if not 0.05 < is_bg.mean() < 0.95:
        return art
    ring = binary_dilation(is_bg) & ~is_bg
    halo = [np.array([255, 255, 255]), np.array([0, 0, 0]),
            bg.astype(np.int64)][int(rng.integers(3))]
    a = float(rng.uniform(0.3, 0.7))
    out = art.astype(np.float32)
    out[ring] = out[ring] * (1 - a) + halo * a
    return np.clip(out, 0, 255).astype(np.uint8)


def dead_cells(art: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scatter single wrong-coloured pixels (editor accidents, speckle)."""
    h, w = art.shape[:2]
    k = max(1, int(h * w * rng.uniform(0.002, 0.01)))
    ys = rng.integers(0, h, k)
    xs = rng.integers(0, w, k)
    out = art.copy()
    if rng.random() < 0.5:
        out[ys, xs] = rng.integers(0, 256, (k, 3))
    else:
        sy = rng.integers(0, h, k)
        sx = rng.integers(0, w, k)
        out[ys, xs] = art[sy, sx]
    return out


def break_outlines(art: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Punch short gaps into contour lines (fill bleeds through)."""
    from scipy.ndimage import maximum_filter, binary_dilation
    a = art.astype(np.float32)
    luma = a.mean(2)
    m = (maximum_filter(luma, 3) - luma) > 40.0
    if m.mean() < 0.01:
        return art
    ys, xs = np.where(m)
    frac = float(rng.uniform(0.05, 0.25))
    k = max(1, int(len(ys) * frac / 2.5))
    sel = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
    gap = np.zeros_like(m)
    gap[ys[sel], xs[sel]] = True
    gap = binary_dilation(gap, iterations=int(rng.integers(1, 4))) & m
    w = (~m).astype(np.float32)
    fill = _gauss(a * w[..., None], 1.5) / (_gauss(w, 1.5)[..., None] + 1e-6)
    out = a.copy()
    if rng.random() < 0.35:
        alpha = float(rng.uniform(0.5, 0.85))
        out[gap] = a[gap] * (1 - alpha) + fill[gap] * alpha
    else:
        out[gap] = fill[gap]
    return np.clip(out, 0, 255).astype(np.uint8)


DAMAGES = {
    "dead_cells": dead_cells,
    "break_outlines": break_outlines,
    "native_aa": native_aa,
    "overquantize": overquantize,
    "alpha_halo": alpha_halo,
}
