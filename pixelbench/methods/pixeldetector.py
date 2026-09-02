"""PixelDetector adapter (bundled port; detect-only + shared reconstruction)."""
from __future__ import annotations

import numpy as np

from .base import Method, Reconstruction, register
from ._common import dominant_downscale
from . import _pixeldetector_impl as impl


@register
class PixelDetector(Method):
    name = "pixeldetector"

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        rgba = image if image.shape[2] == 4 else np.dstack(
            [image, np.full(image.shape[:2], 255, np.uint8)])
        r = impl.detect(rgba)
        cols, rows = int(r["cols"]), int(r["rows"])
        native = dominant_downscale(rgba[..., :3], cols, rows)
        return Reconstruction(native=native, cols=cols, rows=rows)
