"""Colour metric: are the reconstructed pixels the right colour?

The reconstruction is nearest-resampled to the true native size, then every
reconstructed pixel is compared to the intended original pixel. `delta_e` is
the mean CIELAB colour difference (dE76); ~1 is a just-noticeable difference.
`psnr` is the usual signal-to-noise companion.

Note (see docs/METRICS.md): raw colour error over ALL samples flatters
over-segmenting methods, because predicting more cells than truth preserves
detail through the downsize. The report conditions colour on the
exact-resolution subset for a like-for-like read.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Metric, register
from ._util import to_native, to_lab


@register
class Color(Metric):
    name = "color"
    fields = {"delta_e": "lower", "psnr": "higher"}

    def score(self, sample, rec):
        if rec.native is None:
            return {}
        gt = sample.gt
        native = to_native(rec.native, gt.shape[0], gt.shape[1])
        diff = native.astype(np.float64) - gt.astype(np.float64)
        mse = float((diff ** 2).mean())
        psnr = 99.0 if mse <= 1e-9 else 10.0 * math.log10(255.0 ** 2 / mse)
        de = float(np.sqrt(((to_lab(native) - to_lab(gt)) ** 2).sum(-1)).mean())
        return {"delta_e": de, "psnr": psnr}
