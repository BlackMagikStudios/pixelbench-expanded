"""Placement metrics: are the pixels in the right place, and is the detected
grid sitting on the real one?

Two signals, both distinct from colour:

  pixel_match  Nearest-resample the reconstruction to the true native size and
               measure the share of pixels whose colour exactly matches the
               intended original pixel (within a tiny tolerance). Rewards
               getting the right pixel in the right place, not just the right
               average colour.

  grid_align   Uses the exact per-pixel source-coordinate maps U/V from the
               distortion engine. The TRUE pixel-grid cut positions are the
               integer crossings of U (vertical cuts) and V (horizontal cuts).
               We compare them to the method's implied uniform grid and report
               an F1 of matched cut lines. This is independent of colour and
               of getting the cell COUNT exactly right - it asks whether the
               grid lines the method found actually lie on the real grid.
"""
from __future__ import annotations

import numpy as np

from .base import Metric, register
from ._util import to_native


def _true_cuts(coord_line: np.ndarray) -> np.ndarray:
    """Output positions where the source coordinate crosses an integer -
    i.e. the true pixel-grid cuts along this line."""
    f = np.floor(coord_line)
    idx = np.where(np.abs(np.diff(f)) >= 1)[0]
    return idx.astype(np.float64) + 0.5


def _uniform_cuts(length: int, n_cells: int) -> np.ndarray:
    if n_cells <= 1:
        return np.empty(0)
    step = length / n_cells
    return np.arange(1, n_cells) * step


def _f1(true_c: np.ndarray, pred_c: np.ndarray, tol: float) -> float:
    """F1 of cut lines under ONE-TO-ONE matching, so an over-segmenting method
    that sprays extra cut lines cannot inflate recall - every matched pred
    consumes a distinct true cut."""
    if len(true_c) == 0 and len(pred_c) == 0:
        return 1.0
    if len(true_c) == 0 or len(pred_c) == 0:
        return 0.0
    d = np.abs(true_c[:, None] - pred_c[None, :])
    order = np.argsort(d, axis=None)
    nt, npd = d.shape
    t_used = np.zeros(nt, bool)
    p_used = np.zeros(npd, bool)
    matched = 0
    for flat in order:
        i, j = divmod(int(flat), npd)
        if d[i, j] > tol:
            break
        if not t_used[i] and not p_used[j]:
            t_used[i] = p_used[j] = True
            matched += 1
    prec = matched / npd
    rec = matched / nt
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


@register
class Placement(Metric):
    name = "placement"
    fields = {"pixel_match": "higher", "grid_align": "higher"}

    def score(self, sample, rec):
        out = {}
        gt = sample.gt
        gh, gw = gt.shape[:2]

        if rec.native is not None:
            native = to_native(rec.native, gh, gw)
            match = (np.abs(native.astype(np.int16) - gt.astype(np.int16))
                     .max(axis=2) <= 4)
            out["pixel_match"] = float(match.mean())

        cols, rows = rec.size()
        if cols is not None:
            U, V = sample.U, sample.V
            oh, ow = U.shape
            true_x = _true_cuts(U[oh // 2, :])
            true_y = _true_cuts(V[:, ow // 2])
            pred_x = _uniform_cuts(ow, cols)
            pred_y = _uniform_cuts(oh, rows)
            tol_x = max(1.5, 0.25 * (ow / max(cols, 1)))
            tol_y = max(1.5, 0.25 * (oh / max(rows, 1)))
            out["grid_align"] = 0.5 * (_f1(true_x, pred_x, tol_x)
                                       + _f1(true_y, pred_y, tol_y))
        return out
