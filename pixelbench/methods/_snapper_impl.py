"""Faithful Python port of spritefusion-pixel-snapper (baseline).

From the Rust source (spritefusion-pixel-snapper/src/main.rs): k-means
quantize (16 colors) -> luminance [-1,0,1] gradient profiles -> peaks above
0.2*max with min spacing 4 -> median peak spacing per axis -> cross-axis
resolution (ratio > 1.8 takes the smaller, else average) -> greedy walker
(window 0.35*step, snap to max if > 0.5*profile mean) -> per-cell mode of
quantized colors. Stabilization pass simplified to the uniform re-snap on
skewed cell counts (the Rust's main correction).

Source: https://github.com/Hugo-Dz/spritefusion-pixel-snapper (Rust). This is
a faithful Python reimplementation bundled for benchmarking.
"""
import numpy as np


def _quantize(rgba, k=16):
    import cv2
    rgb = rgba[..., :3].reshape(-1, 3).astype(np.float32)
    a = rgba[..., 3].reshape(-1) if rgba.shape[2] == 4 else None
    sel = rgb if a is None else rgb[a > 0]
    if len(sel) == 0:
        return rgba[..., :3].copy()
    if len(sel) > 60000:
        idx = np.random.default_rng(42).choice(len(sel), 60000, replace=False)
        sel = sel[idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.01)
    _, _, centers = cv2.kmeans(sel, min(k, len(np.unique(sel, axis=0))),
                               None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = ((rgb[:, None, :] - centers[None]) ** 2).sum(-1)
    out = centers[np.argmin(d, 1)].reshape(rgba.shape[0], rgba.shape[1], 3)
    return np.rint(out).astype(np.uint8)


def _profiles(q, alpha):
    gray = (0.299 * q[..., 0] + 0.587 * q[..., 1] + 0.114 * q[..., 2]).astype(np.float64)
    if alpha is not None:
        gray = np.where(alpha == 0, 0.0, gray)
    h, w = gray.shape
    col = np.zeros(w)
    col[1:w - 1] = np.abs(gray[:, 2:] - gray[:, :-2]).sum(0)
    row = np.zeros(h)
    row[1:h - 1] = np.abs(gray[2:, :] - gray[:-2, :]).sum(1)
    return col, row


def _estimate_step(profile):
    mx = profile.max()
    if mx <= 0:
        return None
    thr = 0.2 * mx
    peaks = [i for i in range(1, len(profile) - 1)
             if profile[i] > thr and profile[i] > profile[i - 1]
             and profile[i] > profile[i + 1]]
    if len(peaks) < 2:
        return None
    clean = [peaks[0]]
    for p in peaks[1:]:
        if p - clean[-1] > 3:      # peak_distance_filter 4
            clean.append(p)
    if len(clean) < 2:
        return None
    diffs = sorted(np.diff(clean))
    return float(diffs[len(diffs) // 2])


def _walk(profile, step, limit):
    cuts = [0]
    pos = 0.0
    window = max(step * 0.35, 2.0)
    mean = profile.mean()
    while pos < limit:
        target = pos + step
        if target >= limit:
            cuts.append(limit)
            break
        lo = max(int(target - window), int(pos + 1))
        hi = min(int(target + window), limit)
        if hi <= lo:
            pos = target
            continue
        seg = profile[lo:hi]
        j = lo + int(np.argmax(seg))
        if seg.max() > mean * 0.5:
            cuts.append(j)
            pos = float(j)
        else:
            cuts.append(int(target))
            pos = target
    out = sorted(set(min(max(c, 0), limit) for c in cuts) | {0, limit})
    return np.array(out)


def detect(rgba):
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    q = _quantize(rgba)
    col, row = _profiles(q, alpha)
    sx = _estimate_step(col)
    sy = _estimate_step(row)
    if sx and sy:
        ratio = max(sx, sy) / min(sx, sy)
        s = min(sx, sy) if ratio > 1.8 else (sx + sy) / 2
        sx = sy = s
    elif sx:
        sy = sx
    elif sy:
        sx = sy
    else:
        sx = sy = max(min(w, h) / 64.0, 1.0)

    xs = _walk(col, sx, w)
    ys = _walk(row, sy, h)

    # simplified stabilization: uniform re-snap on skewed counts
    cw = w / max(len(xs) - 1, 1)
    rh = h / max(len(ys) - 1, 1)
    if max(cw, rh) / max(min(cw, rh), 1e-9) > 1.8:
        target = min(cw, rh)
        if cw > target * 1.2:
            xs = np.round(np.arange(0, w + target / 2, target)).astype(int)
            xs[-1] = w
        if rh > target * 1.2:
            ys = np.round(np.arange(0, h + target / 2, target)).astype(int)
            ys[-1] = h

    return {"step_x": float(w / (len(xs) - 1)), "step_y": float(h / (len(ys) - 1)),
            "cols": len(xs) - 1, "rows": len(ys) - 1}


def detect_full(rgba):
    """detect() plus the cut arrays + quantized image for resampling."""
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    q = _quantize(rgba)
    col, row = _profiles(q, alpha)
    sx = _estimate_step(col)
    sy = _estimate_step(row)
    if sx and sy:
        ratio = max(sx, sy) / min(sx, sy)
        s = min(sx, sy) if ratio > 1.8 else (sx + sy) / 2
        sx = sy = s
    elif sx:
        sy = sx
    elif sy:
        sx = sy
    else:
        sx = sy = max(min(w, h) / 64.0, 1.0)
    xs = _walk(col, sx, w)
    ys = _walk(row, sy, h)
    cw = w / max(len(xs) - 1, 1)
    rh = h / max(len(ys) - 1, 1)
    if max(cw, rh) / max(min(cw, rh), 1e-9) > 1.8:
        target = min(cw, rh)
        if cw > target * 1.2:
            xs = np.round(np.arange(0, w + target / 2, target)).astype(int)
            xs[-1] = w
        if rh > target * 1.2:
            ys = np.round(np.arange(0, h + target / 2, target)).astype(int)
            ys[-1] = h
    return {"step_x": float(w / (len(xs) - 1)), "step_y": float(h / (len(ys) - 1)),
            "cols": len(xs) - 1, "rows": len(ys) - 1,
            "_xs": xs.tolist(), "_ys": ys.tolist(), "_q": q}


def resample(rgba, r):
    """Per-cell mode of quantized colors (the Rust resample)."""
    q = r.get("_q")
    if q is None:
        q = _quantize(rgba)
    xs = np.array(r["_xs"])
    ys = np.array(r["_ys"])
    ncx, ncy = len(xs) - 1, len(ys) - 1
    h, w = q.shape[:2]
    ix = np.clip(np.searchsorted(xs, np.arange(w), side="right") - 1, 0, ncx - 1)
    iy = np.clip(np.searchsorted(ys, np.arange(h), side="right") - 1, 0, ncy - 1)
    cell = (iy[:, None] * ncx + ix[None, :]).ravel()
    key = ((q[..., 0].astype(np.int64) >> 0) << 16 | \
           (q[..., 1].astype(np.int64) << 8) | q[..., 2].astype(np.int64)).ravel()
    comp = cell * (1 << 24) + key
    uniq, inv = np.unique(comp, return_inverse=True)
    cnt = np.bincount(inv)
    cells_of = uniq >> 24
    order = np.lexsort((cnt, cells_of))
    co = cells_of[order]
    last = np.flatnonzero(np.r_[co[1:] != co[:-1], True])
    win = uniq[order[last]] & ((1 << 24) - 1)
    out = np.zeros((ncx * ncy, 3), np.uint8)
    out[co[last]] = np.stack([(win >> 16) & 255, (win >> 8) & 255,
                              win & 255], 1).astype(np.uint8)
    low = out.reshape(ncy, ncx, 3)
    if rgba.shape[2] == 4:
        a = (rgba[..., 3].reshape(-1) > 127).astype(np.float64)
        asum = np.bincount(cell, weights=a, minlength=ncx * ncy)
        atot = np.maximum(np.bincount(cell, minlength=ncx * ncy), 1)
        mask = (asum / atot > 0.5).reshape(ncy, ncx)
        low = np.dstack([low, (mask * 255).astype(np.uint8)])
    return low
