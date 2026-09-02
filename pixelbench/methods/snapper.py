"""PixelSnapper adapter (bundled faithful port)."""
from __future__ import annotations

import numpy as np

from .base import Method, Reconstruction, register
from . import _snapper_impl as impl


@register
class PixelSnapper(Method):
    name = "snapper"

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        rgba = image if image.shape[2] == 4 else np.dstack(
            [image, np.full(image.shape[:2], 255, np.uint8)])
        r = impl.detect_full(rgba)
        native = impl.resample(rgba, r)
        return Reconstruction(native=np.asarray(native)[..., :3],
                              cols=int(r["cols"]), rows=int(r["rows"]))
