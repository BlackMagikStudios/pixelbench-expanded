"""Palette metrics: did the method recover the right set of colours?

  ncolor_err   Relative error in the number of distinct colours vs the source
               palette. Over-quantizing (too few) or introducing new colours
               (too many, from blur/noise bleed) both hurt.

  palette_de   Mean CIELAB distance from each recovered colour to its nearest
               source colour - are the colours it used actually in the art's
               palette, regardless of count or placement.
"""
from __future__ import annotations

import numpy as np

from .base import Metric, register
from ._util import to_lab, palette_of


@register
class Palette(Metric):
    name = "palette"
    fields = {"ncolor_err": "lower", "palette_de": "lower"}

    def score(self, sample, rec):
        if rec.native is None:
            return {}
        gt_pal = palette_of(sample.gt)
        rc_pal = palette_of(np.asarray(rec.native)[..., :3])
        n_gt = max(len(gt_pal), 1)
        ncolor_err = abs(len(rc_pal) - len(gt_pal)) / n_gt

        gl = to_lab(gt_pal.reshape(1, -1, 3)).reshape(-1, 3)
        rl = to_lab(rc_pal.reshape(1, -1, 3)).reshape(-1, 3)
        d = np.sqrt(((rl[:, None, :] - gl[None, :, :]) ** 2).sum(-1))
        palette_de = float(d.min(axis=1).mean())
        return {"ncolor_err": float(ncolor_err), "palette_de": palette_de}
