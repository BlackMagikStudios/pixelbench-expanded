"""Curated, versioned distortion suite for pixel-bench.

The primitive engine (``engine.py``) can express a huge space of random
damage. A *benchmark* needs the opposite: a small set of named, documented,
reproducible categories so that everyone's numbers are comparable and citable.

This module defines **suite v1**: a fixed list of distortion categories, each a
factory that builds a ``DistortSpec`` from a seeded RNG. The spec for a given
(image, category) is fully deterministic - derived from a hash of the image id,
the category name and the suite version - so two people running suite v1 on the
same image get byte-identical distorted inputs.

A ``severity`` scalar (default 1.0) multiplies the damage amplitudes for anyone
who wants an easier or harsher variant; the upscale factor is severity-
independent so the size-recovery task stays constant.

To evolve the suite, add a new category (append-only) or cut a new version
constant; never silently change an existing category's parameters, or old
results stop being comparable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .engine import DistortSpec
from . import native as _native

SUITE_VERSION = "v1"


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    build: Callable[[np.random.Generator, float], DistortSpec]
    # optional native-art damage applied BEFORE the upscale; the damaged
    # native then becomes the ground truth the method must recover.
    native: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None


def _clampq(q: float) -> int:
    return int(np.clip(round(q), 30, 92))


# --------------------------------------------------------------- factories
# Each factory picks an upscale factor from its own range and sets the damage
# fields it is responsible for. `sev` scales damage amplitude only.


def _clean_nn(rng, sev):
    s = float(rng.choice([2, 3, 4, 5, 6, 8]))
    return DistortSpec(sx=s, sy=s, resample="nearest")


def _fractional(rng, sev):
    s = float(rng.uniform(2.3, 7.7))
    return DistortSpec(sx=s, sy=s, resample="nearest")


def _nonsquare(rng, sev):
    sx = float(rng.uniform(2.5, 7.0))
    # force a visibly different vertical scale (>= 1.3x apart either way)
    sy = float(rng.uniform(2.5, 7.0))
    while 0.77 < sx / sy < 1.3:
        sy = float(rng.uniform(2.5, 7.0))
    return DistortSpec(sx=sx, sy=sy, resample="nearest")


def _soft_bilinear(rng, sev):
    s = float(rng.uniform(3.0, 7.0))
    return DistortSpec(sx=s, sy=s, resample="bilinear", blur=0.5 * sev)


def _soft_bicubic(rng, sev):
    s = float(rng.uniform(3.0, 7.0))
    return DistortSpec(sx=s, sy=s, resample="bicubic",
                       blur=0.3 * sev, sharpen=0.9 * sev)


def _jpeg(rng, sev):
    s = float(rng.uniform(3.0, 7.0))
    resample = "nearest" if rng.random() < 0.5 else "bilinear"
    q = _clampq(80 - 34 * sev * rng.uniform(0.6, 1.0))
    return DistortSpec(sx=s, sy=s, resample=resample, jpeg_quality=q,
                       jpeg_twice=bool(rng.random() < 0.3 * sev))


def _noise(rng, sev):
    s = float(rng.uniform(3.0, 7.0))
    return DistortSpec(sx=s, sy=s, resample="bilinear",
                       noise=6.0 * sev, chroma_noise=5.0 * sev, blur=0.4 * sev)


def _drift(rng, sev):
    s = float(rng.uniform(3.0, 7.0))
    return DistortSpec(sx=s, sy=s, resample="nearest",
                       drift=float(np.clip(0.12 * sev, 0, 0.28)))


def _mush(rng, sev):
    s = float(rng.uniform(3.5, 7.0))
    return DistortSpec(sx=s, sy=s, resample="bilinear",
                       mush=1.6 * sev, blur=0.7 * sev)


def _painterly(rng, sev):
    # painterly texture needs big pseudo-cells to be expressible
    s = float(rng.uniform(6.0, 11.0))
    return DistortSpec(sx=s, sy=s, resample="nearest",
                       cell_gradient=22.0 * sev, cell_noise=9.0 * sev,
                       blur=0.6 * sev, drift=float(np.clip(0.05 * sev, 0, 0.15)))


def _kitchen_sink(rng, sev):
    s = float(rng.uniform(4.0, 8.0))
    return DistortSpec(
        sx=s, sy=s, resample="bilinear",
        drift=float(np.clip(0.10 * sev, 0, 0.22)),
        mush=1.2 * sev, blur=0.9 * sev, sharpen=0.5 * sev,
        noise=6.0 * sev, chroma_noise=6.0 * sev,
        color_field=float(np.clip(0.12 * sev, 0, 0.25)),
        downup=0.8, jpeg_quality=_clampq(70 - 20 * sev))


# ---- helper for the isolated single-distortion categories ----
# A standard modest upscale so the ONE effect under test is what varies.


def _mk(rng, resample="nearest", lo=3.0, hi=6.0, **fields):
    s = float(rng.uniform(lo, hi))
    spec = DistortSpec(sx=s, sy=s, resample=resample)
    for k, v in fields.items():
        setattr(spec, k, v)
    return spec


# geometry (grid) isolates
def _jitter(rng, sev):
    return _mk(rng, jitter=float(np.clip(0.12 * sev, 0, 0.4)))


def _row_jitter(rng, sev):
    return _mk(rng, row_jitter=1.2 * sev)


def _block_reset(rng, sev):
    return _mk(rng, block_reset=int(rng.integers(2, 5)))


def _warp(rng, sev):
    return _mk(rng, warp_amp=2.0 * sev, warp_freq=float(rng.uniform(1.5, 3.5)))


def _subpixel_shift(rng, sev):
    return _mk(rng, subpixel_shift=float(np.clip(0.5 * sev, 0.2, 0.75)))


def _resize_chain(rng, sev):
    return _mk(rng, resize_chain=int(rng.integers(1, 4)))


def _downup(rng, sev):
    return _mk(rng, downup=float(np.clip(0.85 - 0.25 * sev, 0.5, 0.9)))


# resampling kernels
def _bilinear(rng, sev):
    return _mk(rng, resample="bilinear")


def _bicubic(rng, sev):
    return _mk(rng, resample="bicubic")


def _grid_soften(rng, sev):
    return _mk(rng, drift=0.04, grid_soften=0.8 * sev)


# photometric isolates
def _blur(rng, sev):
    return _mk(rng, blur=1.0 * sev)


def _sharpen(rng, sev):
    return _mk(rng, resample="bilinear", sharpen=1.2 * sev)


def _glow(rng, sev):
    return _mk(rng, glow=1.0 * sev)


def _chroma_noise(rng, sev):
    return _mk(rng, chroma_noise=9.0 * sev)


def _color_field(rng, sev):
    return _mk(rng, color_field=float(np.clip(0.15 * sev, 0, 0.25)))


def _banding(rng, sev):
    return _mk(rng, banding=int(np.clip(round(20 - 8 * sev), 6, 32)))


def _median(rng, sev):
    return _mk(rng, median=True)


# codec isolates
def _jpeg_twice(rng, sev):
    q = _clampq(65 - 25 * sev * rng.uniform(0.6, 1.0))
    return _mk(rng, resample="bilinear", jpeg_quality=q, jpeg_twice=True)


def _webp(rng, sev):
    return _mk(rng, webp_quality=_clampq(70 - 30 * sev))


def _chroma_sub(rng, sev):
    return _mk(rng, resample="bilinear", chroma_sub=True)


def _heavy_jpeg(rng, sev):
    return _mk(rng, resample="bilinear",
               jpeg_quality=_clampq(45 - 15 * sev), jpeg_twice=True)


# painterly / cell-texture isolates (need bigger pseudo-cells)
def _cell_gradient(rng, sev):
    return _mk(rng, lo=6.0, hi=10.0, cell_gradient=24.0 * sev)


def _cell_noise(rng, sev):
    return _mk(rng, lo=6.0, hi=10.0, cell_noise=11.0 * sev)


def _cell_texture(rng, sev):
    return _mk(rng, lo=6.0, hi=10.0, cell_texture=float(np.clip(0.22 * sev, 0, 0.35)))


# native-damage categories: a realistic mild upscale carries the damaged art
def _native_carrier(rng, sev):
    return _mk(rng, resample="bilinear", lo=3.5, hi=6.0, blur=0.3)


# extra realistic combinations
def _mush_warp(rng, sev):
    return _mk(rng, resample="bilinear", lo=3.5, hi=6.5,
               mush=1.4 * sev, warp_amp=1.8 * sev, warp_freq=2.5, blur=0.5 * sev)


def _ai_upscale(rng, sev):
    return _mk(rng, resample="bilinear", lo=4.0, hi=7.0, blur=0.6 * sev,
               sharpen=0.9 * sev, glow=0.7 * sev,
               jpeg_quality=_clampq(72 - 12 * sev))


def _screenshot(rng, sev):
    return _mk(rng, resample="bilinear", lo=3.0, hi=5.5,
               resize_chain=int(rng.integers(1, 3)), blur=0.4 * sev,
               noise=3.0 * sev, jpeg_quality=_clampq(74 - 14 * sev))


CATEGORIES = [
    Category("clean_nn",
             "Integer nearest-neighbour upscale, no photometric damage. "
             "The floor case every method should solve.", _clean_nn),
    Category("fractional",
             "Non-integer square upscale (nearest). Cells are uneven integer "
             "widths; recovering the true scale is the challenge.", _fractional),
    Category("nonsquare",
             "Independent horizontal and vertical scales (nearest). Tests "
             "per-axis detection.", _nonsquare),
    Category("soft_bilinear",
             "Bilinear upscale plus light blur - the grid edges are smeared.",
             _soft_bilinear),
    Category("soft_bicubic",
             "Bicubic upscale with unsharp-mask halos, as AI upscalers add.",
             _soft_bicubic),
    Category("jpeg",
             "Upscale then JPEG recompression (blocking + chroma smear), "
             "sometimes double-compressed.", _jpeg),
    Category("noise",
             "Upscale plus luma and chroma noise and slight blur.", _noise),
    Category("drift",
             "Smoothly drifting cell size across the image - the grid is not "
             "uniform.", _drift),
    Category("mush",
             "Heavy AI-upscaler 'mush': smooth sub-cell displacement and soft "
             "focus over a bilinear grid.", _mush),
    Category("painterly",
             "Fake-AI-render look: large pseudo-cells with internal gradient "
             "and blob texture, so cells are not flat.", _painterly),
    Category("kitchen_sink",
             "Everything at once at high severity - drift, mush, blur, "
             "sharpen, noise, colour cast, down-up and JPEG.", _kitchen_sink),

    # --- geometry isolates ---
    Category("jitter", "Per-cut grid jitter: every cell edge offset a little.",
             _jitter),
    Category("row_jitter", "Whole native rows/columns shifted independently.",
             _row_jitter),
    Category("block_reset", "Image split into blocks, each with its own grid "
             "phase (sprite-sheet stitch).", _block_reset),
    Category("warp", "Smooth sinusoidal geometric warp of the grid.", _warp),
    Category("subpixel_shift", "A constant fractional-pixel offset (ghosted "
             "composite).", _subpixel_shift),
    Category("resize_chain", "Several successive resizes at random filters and "
             "non-integer factors.", _resize_chain),
    Category("downup", "Downscaled then upscaled again (bilinear round trip).",
             _downup),

    # --- resampling kernels ---
    Category("bilinear", "Pure bilinear upscale (smeared grid, no extra "
             "damage).", _bilinear),
    Category("bicubic", "Pure bicubic upscale (ringing at edges).", _bicubic),
    Category("grid_soften", "Uneven grid resampled with a smooth filter.",
             _grid_soften),

    # --- photometric isolates ---
    Category("blur", "Gaussian blur only.", _blur),
    Category("sharpen", "Unsharp-mask halos only.", _sharpen),
    Category("glow", "Additive bloom from bright pixels.", _glow),
    Category("chroma_noise", "Per-channel colour noise.", _chroma_noise),
    Category("color_field", "Low-frequency per-channel colour cast.",
             _color_field),
    Category("banding", "Posterization to few levels per channel.", _banding),
    Category("median", "3x3 median filter (denoise-app look).", _median),

    # --- codec isolates ---
    Category("jpeg_twice", "Double JPEG compression.", _jpeg_twice),
    Category("webp", "WebP re-encoding (different chroma smear from JPEG).",
             _webp),
    Category("chroma_sub", "4:2:0 chroma subsampling (video-codec colour "
             "smear).", _chroma_sub),
    Category("heavy_jpeg", "Very low quality, double-compressed JPEG.",
             _heavy_jpeg),

    # --- painterly / in-cell texture isolates ---
    Category("cell_gradient", "Each pseudo-cell gets an internal linear "
             "shading gradient.", _cell_gradient),
    Category("cell_noise", "Sub-cell blob noise inside each pseudo-cell.",
             _cell_noise),
    Category("cell_texture", "Real-texture infusion inside cells (painterly "
             "AI-render look).", _cell_texture),

    # --- native-art damage (damaged native is the ground truth) ---
    Category("dead_cells", "Scattered single wrong-coloured native pixels.",
             _native_carrier, native=_native.dead_cells),
    Category("break_outlines", "Short gaps punched into contour lines.",
             _native_carrier, native=_native.break_outlines),
    Category("native_aa", "The native art is anti-aliased (soft edges).",
             _native_carrier, native=_native.native_aa),
    Category("overquantize", "The native palette is crushed to fewer colours.",
             _native_carrier, native=_native.overquantize),
    Category("alpha_halo", "Bad-composite fringe around the subject.",
             _native_carrier, native=_native.alpha_halo),

    # --- extra realistic combinations ---
    Category("mush_warp", "Mush plus geometric warp (heavy melt).", _mush_warp),
    Category("ai_upscale", "The classic AI upscaler look: soft, sharpened, "
             "glowing, lightly JPEG'd.", _ai_upscale),
    Category("screenshot", "Screenshot-of-a-screenshot: resize chain, noise "
             "and JPEG.", _screenshot),
]

_BY_NAME = {c.name: c for c in CATEGORIES}


def category_names() -> list[str]:
    return [c.name for c in CATEGORIES]


def _seed(image_id: str, category: str, salt: str = "") -> int:
    h = hashlib.sha256(
        f"{SUITE_VERSION}|{category}|{salt}|{image_id}".encode("utf-8")
    ).hexdigest()
    return int(h[:8], 16)


def make_spec(category: str, image_id: str, severity: float = 1.0) -> DistortSpec:
    """Deterministic DistortSpec for (category, image_id) under suite v1.

    image_id should be a stable identifier for the source image (its filename
    is fine). Same inputs -> identical spec -> identical distorted image.
    """
    if category not in _BY_NAME:
        raise KeyError(f"unknown category {category!r}; have {category_names()}")
    rng = np.random.default_rng(_seed(image_id, category, "params"))
    spec = _BY_NAME[category].build(rng, float(severity))
    spec.seed = _seed(image_id, category, "distort")  # engine's internal rng
    spec.name = category
    return spec


def damage_native(category: str, image_id: str, art: np.ndarray) -> np.ndarray:
    """Apply a category's native-art damage (if any) before the upscale.
    Deterministic per (category, image_id). Returns the art unchanged for
    categories with no native damage. The result is the ground truth."""
    cat = _BY_NAME.get(category)
    if cat is None or cat.native is None:
        return art
    rng = np.random.default_rng(_seed(image_id, category, "native"))
    try:
        return cat.native(art, rng)
    except Exception:
        return art  # a damage that can't apply (e.g. no outlines) is a no-op
