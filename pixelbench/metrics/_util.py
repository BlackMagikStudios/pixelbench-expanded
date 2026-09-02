"""Shared helpers for metrics."""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_native(native: np.ndarray, gt_h: int, gt_w: int) -> np.ndarray:
    """Nearest-resample a reconstruction to the true native size so every
    reconstructed pixel lines up with the intended original pixel."""
    native = np.asarray(native)[..., :3].astype(np.uint8)
    if native.shape[:2] != (gt_h, gt_w):
        native = np.array(Image.fromarray(native).resize((gt_w, gt_h),
                                                          Image.NEAREST))
    return native


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIELAB (L in [0,100], a,b in ~[-127,127]) via cv2."""
    import cv2
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def palette_of(rgb: np.ndarray, cap: int = 64) -> np.ndarray:
    """Distinct colours of an image as an (N,3) uint8 array. If there are more
    than `cap`, reduce to `cap` representative colours by k-means."""
    flat = rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    if len(uniq) <= cap:
        return uniq.astype(np.uint8)
    import cv2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    sel = flat.astype(np.float32)
    if len(sel) > 40000:
        sel = sel[np.random.default_rng(0).choice(len(sel), 40000, replace=False)]
    _, _, centers = cv2.kmeans(sel, cap, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return np.rint(centers).astype(np.uint8)
