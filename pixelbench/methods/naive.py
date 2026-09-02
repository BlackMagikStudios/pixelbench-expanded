"""Naive floor baseline: guess the scale from the modal constant-colour run.

The crudest real detector - look at how long identical pixels run before the
colour changes, take the most common run length as the cell size, and
dominant-downscale on that uniform grid. Works on clean nearest-neighbour
upscales, collapses the moment the grid is blurred or noisy. Included as the
zero-effort floor every real method should beat.
"""
from __future__ import annotations

import numpy as np

from .base import Method, Reconstruction, register
from ._common import dominant_downscale


def _modal_run(gray: np.ndarray, axis: int) -> float:
    """Most common run length of constant colour along an axis (vectorized).

    A run starts at each colour change, plus a forced break at the start of
    every row/column so runs never cross the image edge. Run lengths are the
    gaps between consecutive run starts in row-major order."""
    g = gray if axis == 1 else gray.T
    chg = np.ones(g.shape, bool)
    chg[:, 1:] = g[:, 1:] != g[:, :-1]      # colour change within the line
    flat = chg.ravel()
    starts = np.flatnonzero(flat)
    if len(starts) == 0:
        return 1.0
    runs = np.diff(np.append(starts, flat.size))
    vals, cnts = np.unique(runs, return_counts=True)
    return float(vals[np.argmax(cnts)])


@register
class Naive(Method):
    name = "naive"

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        rgb = image[..., :3]
        gray = rgb.astype(np.int32).sum(2)
        sx = max(_modal_run(gray, 1), 1.0)
        sy = max(_modal_run(gray, 0), 1.0)
        h, w = rgb.shape[:2]
        cols = max(1, round(w / sx))
        rows = max(1, round(h / sy))
        native = dominant_downscale(rgb, cols, rows)
        return Reconstruction(native=native, cols=cols, rows=rows)
