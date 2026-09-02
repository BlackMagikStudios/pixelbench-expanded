"""Shared reconstruction helpers for method adapters."""
from __future__ import annotations

import numpy as np


def quantize_kmeans(rgb: np.ndarray, k: int = 16) -> np.ndarray:
    """Global k-means colour quantization (cv2)."""
    import cv2
    flat = rgb.reshape(-1, 3).astype(np.float32)
    uniq = min(k, len(np.unique(flat, axis=0)))
    if uniq < 2:
        return rgb.copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.5)
    sel = flat
    if len(flat) > 40000:
        idx = np.random.default_rng(0).choice(len(flat), 40000, replace=False)
        sel = flat[idx]
    _, _, centers = cv2.kmeans(sel, uniq, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = ((flat[:, None, :] - centers[None]) ** 2).sum(-1)
    return np.rint(centers[np.argmin(d, 1)]).astype(np.uint8).reshape(rgb.shape)


def dominant_downscale(rgb: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Per-cell dominant colour on a uniform cols x rows grid. The shared
    reconstruction for detect-only methods."""
    h, w = rgb.shape[:2]
    cols = max(1, int(cols)); rows = max(1, int(rows))
    q = quantize_kmeans(rgb)
    ix = np.clip((np.arange(w) * cols // w), 0, cols - 1)
    iy = np.clip((np.arange(h) * rows // h), 0, rows - 1)
    cell = (iy[:, None] * cols + ix[None, :]).ravel()
    key = (q[..., 0].astype(np.int64) << 16 | q[..., 1].astype(np.int64) << 8
           | q[..., 2].astype(np.int64)).ravel()
    comp = cell.astype(np.int64) * (1 << 24) + key
    uniq, inv = np.unique(comp, return_inverse=True)
    cnt = np.bincount(inv)
    cells_of = uniq >> 24
    order = np.lexsort((cnt, cells_of))
    co = cells_of[order]
    last = np.flatnonzero(np.r_[co[1:] != co[:-1], True])
    win = uniq[order[last]] & ((1 << 24) - 1)
    out = np.zeros((cols * rows, 3), np.uint8)
    out[co[last]] = np.stack([(win >> 16) & 255, (win >> 8) & 255,
                              win & 255], 1).astype(np.uint8)
    return out.reshape(rows, cols, 3)
