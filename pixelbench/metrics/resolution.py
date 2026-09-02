"""Resolution metric: did the method recover the true native size?

The single hardest and most important signal - every downstream pixel depends
on getting cols x rows right.
"""
from __future__ import annotations

from .base import Metric, register


@register
class Resolution(Metric):
    name = "resolution"
    fields = {
        "exact": "higher",       # both dims exactly right (0/1)
        "within1": "higher",     # both within +-1 cell (0/1)
        "within5pct": "higher",  # both within 5% (min 1 cell) (0/1)
        "rel_err": "lower",      # mean(|dcols|/W, |drows|/H)
    }

    def score(self, sample, rec):
        cols, rows = rec.size()
        if cols is None:
            return {"exact": 0.0, "within1": 0.0, "within5pct": 0.0,
                    "rel_err": 1.0}
        w, h = sample.gt_cols, sample.gt_rows
        dc, dr = abs(cols - w), abs(rows - h)
        tolc = max(1, round(0.05 * w))
        tolr = max(1, round(0.05 * h))
        return {
            "exact": float(cols == w and rows == h),
            "within1": float(dc <= 1 and dr <= 1),
            "within5pct": float(dc <= tolc and dr <= tolr),
            "rel_err": 0.5 * (dc / w + dr / h),
        }
