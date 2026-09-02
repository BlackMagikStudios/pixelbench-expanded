"""Bundled port of PixelDetector's grid detection (all-peaks median spacing).

Source: https://github.com/Astropulse/pixeldetector (MIT). Detection only;
pixel-bench reconstructs the native image via a shared dominant-colour
downscale on the detected grid.
"""
import numpy as np
from scipy.signal import find_peaks


def detect(rgba):
    npim = rgba[..., :3].astype(np.float64)
    hdiff = np.sqrt(((npim[:, :-1] - npim[:, 1:]) ** 2).sum(2)).sum(0)
    vdiff = np.sqrt(((npim[:-1] - npim[1:]) ** 2).sum(2)).sum(1)
    hp, _ = find_peaks(hdiff, distance=1, height=0.0)
    vp, _ = find_peaks(vdiff, distance=1, height=0.0)
    sx = float(np.median(np.diff(hp))) if len(hp) > 1 else 1.0
    sy = float(np.median(np.diff(vp))) if len(vp) > 1 else 1.0
    h, w = rgba.shape[:2]
    return {"step_x": sx, "step_y": sy,
            "cols": max(1, round(w / sx)), "rows": max(1, round(h / sy))}


if __name__ == "__main__":
    from tools.bench_common import run_bench
    run_bench(detect, "baseline_pixeldetector")
