"""pixel-bench smoke tests. Run with `pytest` or `python tests/test_pixelbench.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelbench.distort import (make_spec, distort, category_names,
                                damage_native)
from pixelbench.methods import all_methods, Reconstruction
from pixelbench.metrics import all_metrics, field_directions, Sample


def _art(h=24, w=20, ncolors=6, seed=1):
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 256, (ncolors, 3), np.uint8)
    return pal[rng.integers(0, ncolors, (h, w))].astype(np.uint8)


def test_suite_is_deterministic():
    for cat in category_names():
        a = make_spec(cat, "img.png")
        b = make_spec(cat, "img.png")
        assert (a.sx, a.sy, a.seed, a.resample) == (b.sx, b.sy, b.seed, b.resample)


def test_suite_covers_all_distortion_types():
    # the full suite exercises geometry, kernels, photometric, codec, cell
    # texture, native-art damage, and combinations
    cats = set(category_names())
    assert len(cats) >= 43
    for expected in ("dead_cells", "break_outlines", "native_aa", "overquantize",
                     "alpha_halo", "warp", "glow", "banding", "webp", "chroma_sub",
                     "cell_texture", "resize_chain", "subpixel_shift"):
        assert expected in cats, expected


def test_native_damage_is_deterministic_and_changes_art():
    art = _art(seed=5)
    for cat in ("dead_cells", "overquantize"):
        a = damage_native(cat, "img.png", art.copy())
        b = damage_native(cat, "img.png", art.copy())
        assert np.array_equal(a, b)               # deterministic
        assert a.shape == art.shape
    # a non-damage category returns the art untouched
    assert np.array_equal(damage_native("clean_nn", "img.png", art.copy()), art)


def test_every_category_distorts_with_uv():
    art = _art()
    for cat in category_names():
        spec = make_spec(cat, "img.png")
        o = distort(art, spec, np.random.default_rng(spec.seed))
        assert o["img"].dtype == np.uint8 and o["img"].ndim == 3
        assert o["U"].shape == o["img"].shape[:2] == o["V"].shape


def test_metric_directions_cover_all_fields():
    dirs = field_directions()
    for m in all_metrics():
        for f, d in m.fields.items():
            assert d in ("higher", "lower")
            assert dirs[f] == d


def test_perfect_reconstruction_scores_perfect():
    art = _art()
    spec = make_spec("clean_nn", "img.png")
    o = distort(art, spec, np.random.default_rng(spec.seed))
    rgba = np.dstack([o["img"], np.full(o["img"].shape[:2], 255, np.uint8)])
    sample = Sample(gt=art, distorted=rgba, U=o["U"], V=o["V"],
                    gt_cols=art.shape[1], gt_rows=art.shape[0],
                    category="clean_nn", severity=1.0, image_id="img.png")
    perfect = Reconstruction(native=art, cols=art.shape[1], rows=art.shape[0])
    scores = {}
    for met in all_metrics():
        scores.update(met.score(sample, perfect))
    assert scores["exact"] == 1.0
    assert scores["delta_e"] < 1e-6
    assert scores["pixel_match"] == 1.0
    assert scores["grid_align"] == 1.0


def test_grid_align_penalizes_oversegmentation():
    # a reconstruction claiming 4x too many cells must not score high grid_align
    art = _art()
    spec = make_spec("clean_nn", "img.png")
    o = distort(art, spec, np.random.default_rng(spec.seed))
    rgba = np.dstack([o["img"], np.full(o["img"].shape[:2], 255, np.uint8)])
    sample = Sample(gt=art, distorted=rgba, U=o["U"], V=o["V"],
                    gt_cols=art.shape[1], gt_rows=art.shape[0],
                    category="clean_nn", severity=1.0, image_id="img.png")
    over = Reconstruction(cols=art.shape[1] * 4, rows=art.shape[0] * 4)
    from pixelbench.metrics.placement import Placement
    ga = Placement().score(sample, over)["grid_align"]
    assert ga < 0.6, ga


def test_registries_nonempty():
    assert {"naive", "snapper", "pixeldetector"} <= {m.name for m in all_methods(only_available=False)}
    assert {"resolution", "color", "placement", "palette"} <= {m.name for m in all_metrics()}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")
