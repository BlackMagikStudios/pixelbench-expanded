"""Benchmark runner: dataset x suite x methods x metrics.

Builds each distorted input once (shared across methods), runs every requested
method on it, scores every metric, and writes a standardized results JSON.

Runs methods in a process pool at below-normal priority and caps the distorted
image size, so a big dataset does not swamp the machine.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .data import list_images, load_art
from .distort import (make_spec, distort, category_names, SUITE_VERSION,
                      damage_native)
from .distort.suite import _seed as _suite_seed
from .methods import all_methods, get_method
from .metrics import all_metrics, Sample, field_directions

MAX_DISTORTED_PX = 450_000
MIN_DISTORTED_SIDE = 18
MAX_SOURCE_SIDE = 384   # native pixel art rarely exceeds this; drops the slow
                        # tail of oversized sources

# per-worker caches
_METHOD_CACHE: dict | None = None
_METRIC_CACHE: list | None = None


def _worker_init():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _methods(names):
    global _METHOD_CACHE
    if _METHOD_CACHE is None:
        _METHOD_CACHE = {m.name: m for m in all_methods()}
    return [(n, _METHOD_CACHE[n]) for n in names if n in _METHOD_CACHE]


def _metrics():
    global _METRIC_CACHE
    if _METRIC_CACHE is None:
        _METRIC_CACHE = all_metrics()
    return _METRIC_CACHE


def _clamp(art_w, art_h, spec):
    px = (art_w * spec.sx) * (art_h * spec.sy)
    if px > MAX_DISTORTED_PX:
        f = math.sqrt(MAX_DISTORTED_PX / px)
        spec.sx = max(spec.sx * f, MIN_DISTORTED_SIDE / art_w)
        spec.sy = max(spec.sy * f, MIN_DISTORTED_SIDE / art_h)
    return spec


def process_task(task):
    """task: (path, category, severity, method_names)."""
    path, category, severity, method_names = task
    image_id = os.path.basename(path)
    try:
        art = load_art(path, np.random.default_rng(_suite_seed(image_id, "bg")))
        art = damage_native(category, image_id, art)   # native damage -> new GT
        h, w = art.shape[:2]
        spec = _clamp(w, h, make_spec(category, image_id, severity))
        o = distort(art, spec, np.random.default_rng(spec.seed))
        img = o["img"]
        if min(img.shape[:2]) < MIN_DISTORTED_SIDE:
            return {"image": image_id, "category": category, "skip": "too small"}
        rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        sample = Sample(gt=art, distorted=rgba, U=o["U"], V=o["V"],
                        gt_cols=w, gt_rows=h, category=category,
                        severity=severity, image_id=image_id)
    except Exception as e:
        return {"image": image_id, "category": category,
                "skip": f"prep error: {e}"}

    rec = {"image": image_id, "category": category, "severity": severity,
           "gt_cols": w, "gt_rows": h, "dist_w": int(img.shape[1]),
           "dist_h": int(img.shape[0]), "methods": {}}
    metrics = _metrics()
    for name, method in _methods(method_names):
        t0 = time.time()
        try:
            r = method.reconstruct(rgba.copy())
            scores = {}
            for met in metrics:
                scores.update(met.score(sample, r))
            scores = {k: round(float(v), 5) for k, v in scores.items()}
            scores["sec"] = round(time.time() - t0, 3)
        except Exception as e:
            scores = {"error": str(e)[:150], "sec": round(time.time() - t0, 3)}
        rec["methods"][name] = scores
    return rec


def build_tasks(paths, categories, severity, method_names):
    return [(p, c, severity, method_names) for p in paths for c in categories]


def aggregate(results, method_names):
    """Per-(method, category) and per-(method, overall) mean of each field.
    Booleans/rates -> mean; error-like fields -> median (robust to outliers)."""
    directions = field_directions()
    median_fields = {"rel_err", "delta_e", "psnr", "palette_de", "sec"}
    scored = [r for r in results if "methods" in r]
    cats = sorted({r["category"] for r in scored})
    agg = {}
    for m in method_names:
        agg[m] = {}
        for scope in cats + ["overall"]:
            rows = [r["methods"].get(m, {}) for r in scored
                    if (scope == "overall" or r["category"] == scope)]
            rows = [x for x in rows if x and "error" not in x]
            if not rows:
                agg[m][scope] = {"n": 0}
                continue
            entry = {"n": len(rows)}
            keys = set().union(*[x.keys() for x in rows]) - {"error"}
            for k in keys:
                vals = [x[k] for x in rows if k in x]
                if not vals:
                    continue
                entry[k] = (float(np.median(vals)) if k in median_fields
                            else float(np.mean(vals)))
            agg[m][scope] = entry
    return {"categories": cats, "directions": directions, "by_method": agg}


def _flush_print(*a):
    print(*a, flush=True)


def run(data_dir, out_path, methods=None, categories=None, severity=1.0,
        limit=None, workers=6, seed=1234, progress=_flush_print):
    paths = list_images(data_dir, max_side=MAX_SOURCE_SIDE)
    if not paths:
        raise SystemExit(f"no usable images in {data_dir}")
    if limit and limit < len(paths):
        idx = np.random.default_rng(seed).choice(len(paths), limit, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    cats = categories or category_names()
    avail = {m.name for m in all_methods()}
    mnames = [m for m in (methods or sorted(avail)) if m in avail]
    if methods:
        missing = [m for m in methods if m not in avail]
        if missing:
            progress(f"skipping unavailable methods: {missing}")
    tasks = build_tasks(paths, cats, severity, mnames)
    progress(f"pixel-bench suite {SUITE_VERSION} | {len(paths)} images x "
             f"{len(cats)} categories = {len(tasks)} inputs | "
             f"methods: {mnames} | workers {workers}")

    results = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init) as ex:
        futs = [ex.submit(process_task, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % max(1, len(tasks) // 40) == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(tasks) - done) / rate if rate else 0
                progress(f"  {done}/{len(tasks)}  {el:.0f}s  {rate:.1f}/s  "
                         f"ETA {eta:.0f}s")

    import json
    from . import __version__
    scored = [r for r in results if "methods" in r]
    payload = {
        "pixelbench_version": __version__,
        "suite": SUITE_VERSION,
        "severity": severity,
        "seed": seed,
        "dataset": {"path": os.path.abspath(data_dir), "n_images": len(paths)},
        "methods": mnames,
        "n_inputs": len(tasks),
        "n_scored": len(scored),
        "wall_s": round(time.time() - t0, 1),
        "aggregate": aggregate(results, mnames),
        "samples": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    progress(f"done: {len(scored)} scored, {len(results)-len(scored)} skipped, "
             f"{time.time()-t0:.0f}s -> {out_path}")
    return payload
