"""Dataset loading for pixel-bench.

pixel-bench does not ship a corpus - you supply your own folder of **native
(1x) pixel art**: images where one stored pixel is one art pixel, with no
upscaling, blur, or resampling. Everything downstream assumes this.

`load_art` decodes an image to RGB (compositing transparency over a flat
colour, the way it appears before distortion). `list_images` finds candidate
files. `looks_upscaled` is a cheap heuristic used by `pixelbench validate` to
warn when an input is probably NOT native 1x (e.g. someone pointed it at
already-upscaled art), because that quietly corrupts every score.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".bmp", ".gif", ".webp")


def list_images(folder: str, min_side: int = 8, max_side: int = 1024,
                stills_only: bool = True) -> list[str]:
    """Absolute paths of candidate 1x images in `folder` (non-recursive by
    default; pass a folder of pixel art). Filters by extension and size."""
    out = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(folder, fn)
        try:
            with Image.open(path) as im:
                w, h = im.size
                if stills_only and getattr(im, "n_frames", 1) > 1:
                    continue
        except Exception:
            continue
        if min_side <= w <= max_side and min_side <= h <= max_side:
            out.append(path)
    return out


def load_art(path: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Load native pixel art as RGB uint8. Transparent art is composited over
    a random flat colour (how it appears in the wild before distortion)."""
    if rng is None:
        rng = np.random.default_rng(0)
    im = Image.open(path)
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    a = np.array(im)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 4:
        if (a[:, :, 3] < 255).any():
            bg = rng.integers(0, 256, 3)
            alpha = a[:, :, 3:4].astype(np.float32) / np.float32(255.0)
            rgb = a[:, :, :3].astype(np.float32) * alpha + \
                bg.astype(np.float32) * (1 - alpha)
            return np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(a[:, :, :3])
    return np.ascontiguousarray(a[:, :, :3])


def looks_upscaled(path: str) -> tuple[bool, str]:
    """Cheap heuristic: does this image look like it is NOT native 1x?

    True (with a reason) if it appears upscaled - i.e. it has large runs of
    identical pixels forming a regular block grid, which native pixel art of
    this size would not. Not authoritative; used only to warn contributors
    that their dataset may be pre-scaled.
    """
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        return True, f"could not open ({e})"
    a = np.asarray(im)
    h, w = a.shape[:2]
    if min(h, w) < 8:
        return False, ""
    # Modal horizontal/vertical run length of identical adjacent pixels.
    def modal_run(arr, axis):
        same = np.all(arr[:, 1:] == arr[:, :-1], axis=2) if axis == 1 \
            else np.all(arr[1:, :] == arr[:-1, :], axis=2)
        # run lengths along the axis
        runs = []
        line_iter = same if axis == 1 else same.T
        for line in line_iter:
            c = 1
            for v in line:
                if v:
                    c += 1
                else:
                    runs.append(c); c = 1
            runs.append(c)
        if not runs:
            return 1.0
        vals, cnts = np.unique(runs, return_counts=True)
        return float(vals[np.argmax(cnts)])
    rx = modal_run(a, 1)
    ry = modal_run(a, 0)
    # If the most common run length on both axes is >= 3 and roughly equal,
    # the image is very likely an integer upscale of smaller art.
    if rx >= 3 and ry >= 3 and 0.5 <= rx / ry <= 2.0:
        return True, (f"modal pixel run ~{rx:.0f}x{ry:.0f}; looks like an "
                      f"integer upscale, not native 1x")
    return False, ""
