"""unfake.py adapter (optional dependency: `pip install unfake`).

unfake detects a single integer, square scale factor and downscales by it, so
it cannot express non-integer or non-square native grids - a structural limit
worth keeping in mind when reading its scores.
"""
from __future__ import annotations

import logging

import numpy as np

from .base import Method, Reconstruction, register

# unfake logs a handful of INFO lines per image; silence them so a benchmark
# run over thousands of inputs stays readable.
for _name in ("unfake", "unfake.py"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False


@register
class Unfake(Method):
    name = "unfake"
    requires = ("unfake",)

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        import unfake
        # belt-and-suspenders: some versions attach the logger at call time
        logging.getLogger("unfake.py").setLevel(logging.ERROR)
        rgba = image if image.shape[2] == 4 else np.dstack(
            [image, np.full(image.shape[:2], 255, np.uint8)])
        res = unfake.process_image_sync(
            rgba, max_colors=32, detect_method="auto", downscale_method="dominant")
        arr = np.asarray(res["image_array"])
        return Reconstruction(native=arr[..., :3],
                              cols=int(arr.shape[1]), rows=int(arr.shape[0]))
