"""Pixel Art Fixer adapter (optional dependency: the public `pixelfixer`
package - https://github.com/Retro-Diffusion/pixel-art-fixer).

The algorithm (non-ML) fixer by Retro Diffusion: multi-signal grid detection
with consensus/arbitration, then two-stage packing (quantize-for-structure +
original-pixels-for-color, adaptive palette). Supports non-square steps.

Set PIXELFIXER_RUST to the native binary path to run the (much faster) Rust
core instead of the Python reference; otherwise the Python package is used.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

from .base import Method, Reconstruction, register


@register
class PixelFixer(Method):
    name = "fixer"
    requires = ("pixelfixer",)

    @classmethod
    def available(cls) -> bool:  # allow the Rust-binary path without the pkg
        if os.environ.get("PIXELFIXER_RUST"):
            return True
        return super().available()

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        rgba = image if image.shape[2] == 4 else np.dstack(
            [image, np.full(image.shape[:2], 255, np.uint8)])

        exe = os.environ.get("PIXELFIXER_RUST")
        if exe:
            with tempfile.TemporaryDirectory() as d:
                ip = os.path.join(d, "in.png")
                op = os.path.join(d, "out.png")
                Image.fromarray(rgba).save(ip)
                cmd = [exe, "process", ip, op, "full"]
                if os.environ.get("PIXELFIXER_LEGACY"):
                    cmd.append("legacy")     # A/B: old grid-cut reconstruction
                out = subprocess.run(cmd, capture_output=True, text=True)
                meta = json.loads(out.stdout.strip().splitlines()[-1])
                native = np.asarray(Image.open(op).convert("RGB"))
            return Reconstruction(native=native,
                                  cols=int(meta["cols"]), rows=int(meta["rows"]))

        from pixelfixer.api import process
        try:
            r = process(rgba, mode="full", return_png=False)
            native = np.asarray(r["array"])[..., :3]
        except TypeError:
            buf = io.BytesIO()
            Image.fromarray(rgba).save(buf, "PNG")
            r = process(buf.getvalue())
            native = np.asarray(Image.open(io.BytesIO(r["png"])).convert("RGB"))
        return Reconstruction(native=native,
                              cols=int(r["cols"]), rows=int(r["rows"]))
