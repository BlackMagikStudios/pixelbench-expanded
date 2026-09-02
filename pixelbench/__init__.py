"""pixel-bench: an open benchmark for pixel-art reconstruction.

Point it at a folder of native (1x) pixel art. It distorts each image through
a curated, versioned suite of real-world damage, runs any number of pluggable
reconstruction methods, and scores them on the signals that matter: correct
native size, correct colors, correct pixel placement, and grid alignment.

Quickstart:
    pixelbench run --data ./my_pixel_art --out results/run.json
    pixelbench report results/run.json

Add your own method or metric in a single file (see CONTRIBUTING.md).
"""

__version__ = "0.2.0a1"
