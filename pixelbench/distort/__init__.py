"""Distortion layer: primitive engine + curated versioned suite."""
from .engine import DistortSpec, distort, sample_spec, spec_dict  # noqa: F401
from .suite import (  # noqa: F401
    SUITE_VERSION,
    CATEGORIES,
    category_names,
    make_spec,
    damage_native,
)
