"""pixel-bench distortion engine: primitive operations.

Takes native (1x) pixel art and applies the kinds of damage that turn real
pixel art into fake pixel art in the wild: fractional/non-square upscales,
smooth grid drift, per-row and per-block phase shifts, warps, mush, in-cell
painterly texture, and a full photometric chain (blur, sharpen, noise, jpeg,
webp, down-up, banding, glow).

The distinguishing feature: every distortion returns the EXACT per-pixel
source-coordinate maps (U, V). For each output pixel, U/V give the continuous
coordinate of the native source pixel it came from, in native pixel units
(source pixel i occupies u in [i, i+1)). Integer crossings of U/V are the
true pixel-grid cut positions, so the maps are dense ground truth for scoring
grid alignment and pixel placement of a reconstruction — including under
drift, jitter, row-jitter, block phase resets and warps. Photometric
distortions (blur, sharpen, noise, chroma noise, color fields, posterize,
jpeg, down-up resampling) leave U/V unchanged.

This module is the primitive layer. The curated, versioned, reproducible
benchmark categories built on top of it live in ``suite.py``.

Pure numpy / scipy / PIL (optional cv2 acceleration). No other dependencies.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
    cv2.setNumThreads(0)          # workers are process-parallel already
except Exception:                 # incl. numpy-ABI import breakage
    cv2 = None


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, cv2-accelerated (3-10x scipy), 2D or HxWxC."""
    if cv2 is not None:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
    from scipy.ndimage import gaussian_filter
    if img.ndim == 3:
        return gaussian_filter(img, sigma=(sigma, sigma, 0))
    return gaussian_filter(img, sigma=sigma)


# ----------------------------------------------------------------- spec


@dataclass
class DistortSpec:
    # --- geometry ----------------------------------------------------
    sx: float = 4.0              # horizontal upscale factor (can be fractional)
    sy: float = 4.0              # vertical upscale factor (non-square when != sx)
    resample: str = "nearest"    # nearest | bilinear | bicubic (affine path)
    drift: float = 0.0           # smooth cell-size drift, fraction of cell
    jitter: float = 0.0          # per-cut jitter, fraction of cell (0..0.45)
    row_jitter: float = 0.0      # per-native-row/col independent shift, output px
    block_reset: int = 0         # NxN blocks with independent grid phase
    warp_amp: float = 0.0        # smooth sinusoidal warp, output px
    warp_freq: float = 2.0
    mush: float = 0.0            # random smooth sub-cell displacement, output px
    # --- cell texture (painterly AI fake look) ------------------------
    cell_gradient: float = 0.0   # per-cell random linear gradient amplitude
                                 # (0-255 units; cells get internal shading)
    cell_noise: float = 0.0      # sub-cell blob noise amplitude (0-255);
                                 # structured texture INSIDE pseudo-pixels
    cell_texture: float = 0.0    # multiplicative REAL-texture infusion
                                 # amplitude (0..0.5); texture array is
                                 # supplied by the caller (e.g. crops of
                                 # real AI renders) — the painterly look
    # --- photometric -------------------------------------------------
    grid_soften: float = 0.0     # gaussian blur emulating a smooth resampler
                                 # on non-affine (jittered/drifted) grids
    resize_chain: int = 0        # N extra resizes with random filters and
                                 # non-integer factors (screenshot-of-a-
                                 # screenshot chains; BSRGAN down-up lesson)
    median: bool = False         # 3x3 median (denoise-app look)
    webp_quality: Optional[int] = None   # webp re-encode (chroma smearing
                                         # different from jpeg)
    downup: float = 0.0          # 0=off, else 0.5..0.95: bilinear down then up
    chroma_sub: bool = False     # 4:2:0 chroma subsampling (video-codec
                                 # color smear; screenshots of videos)
    glow: float = 0.0            # additive bloom from bright pixels
                                 # (AI-upscaler / 'enhance' filter look)
    subpixel_shift: float = 0.0  # constant 0.2-0.7px resample offset
                                 # (ghost edges from a bad composite)
    blur: float = 0.0            # gaussian blur sigma (heavy allowed)
    sharpen: float = 0.0         # unsharp-mask amount (AI-upscaler halos)
    noise: float = 0.0           # luma-correlated gaussian noise sigma (0-255)
    chroma_noise: float = 0.0    # per-channel independent noise sigma
    color_field: float = 0.0     # low-freq per-channel gain amplitude (0..0.25)
    banding: int = 0             # posterize to N levels per channel (0=off)
    jpeg_quality: Optional[int] = None
    jpeg_twice: bool = False     # double-compressed jpeg
    seed: int = 0
    name: str = ""


# ------------------------------------------------------- geometry core


def _cuts(n: int, out_len: float, drift: float, jitter_frac: float,
          rng: np.random.Generator) -> np.ndarray:
    """Monotone cut positions for n cells across [0, out_len]."""
    base = out_len / n
    t = np.arange(n) / max(n - 1, 1)
    if drift > 0:
        ph = rng.uniform(0, 2 * np.pi, 2)
        sizes = base * (1.0 + drift * (0.7 * np.sin(2 * np.pi * 1.3 * t + ph[0])
                                       + 0.3 * np.sin(2 * np.pi * 3.1 * t + ph[1])))
    else:
        sizes = np.full(n, base)
    cuts = np.concatenate([[0.0], np.cumsum(sizes)])
    cuts *= out_len / cuts[-1]
    if jitter_frac > 0:
        j = jitter_frac * base
        offs = rng.uniform(-j, j, n + 1)
        offs[0] = offs[-1] = 0.0
        cuts = np.maximum.accumulate(cuts + offs)
        cuts[-1] = out_len
    # enforce strictly increasing (degenerate cells break u interpolation)
    eps = 1e-3
    for _ in range(2):
        bad = np.diff(cuts) < eps
        if not bad.any():
            break
        idx = np.where(bad)[0]
        cuts[idx + 1] = cuts[idx] + eps
        cuts = np.minimum(cuts, out_len)
        cuts[-1] = out_len
    return cuts


def _u_of_x(cuts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Continuous source coordinate u at output positions x (px centers).
    Piecewise linear; u crosses integer i exactly at cuts[i]."""
    n = len(cuts) - 1
    i = np.clip(np.searchsorted(cuts, x, side="right") - 1, 0, n - 1)
    left = cuts[i]
    width = cuts[i + 1] - left
    return i + (x - left) / np.maximum(width, 1e-9)


_PIL_RESAMPLE = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def _render_geometry(art: np.ndarray, spec: DistortSpec,
                     rng: np.random.Generator):
    """Apply the grid-geometry part. Returns (img_f32, U, V) where img is
    float32 0..255 (h_out, w_out, 3) and U/V are float32 source coords."""
    h, w = art.shape[:2]
    ow = max(int(round(w * spec.sx)), 2)
    oh = max(int(round(h * spec.sy)), 2)

    uneven = (spec.drift > 0 or spec.jitter > 0 or spec.row_jitter > 0
              or spec.block_reset > 0)

    if not uneven and spec.resample in _PIL_RESAMPLE:
        # pure affine smooth resize: use PIL for authentic filtering
        pil = Image.fromarray(art)
        img = np.asarray(pil.resize((ow, oh), _PIL_RESAMPLE[spec.resample]),
                         dtype=np.float32)
        u = ((np.arange(ow, dtype=np.float64) + 0.5) * (w / ow))
        v = ((np.arange(oh, dtype=np.float64) + 0.5) * (h / oh))
        U = np.broadcast_to(u[None, :], (oh, ow)).astype(np.float32).copy()
        V = np.broadcast_to(v[:, None], (oh, ow)).astype(np.float32).copy()
        return img, U, V

    xcuts = _cuts(w, ow, spec.drift, spec.jitter, rng)
    ycuts = _cuts(h, oh, spec.drift, spec.jitter, rng)

    X = np.arange(ow, dtype=np.float64) + 0.5
    Y = np.arange(oh, dtype=np.float64) + 0.5

    if spec.row_jitter > 0 or spec.block_reset > 0:
        # per-native-row/col shifts + per-block phase resets (2D shift fields)
        row_off = (rng.uniform(-spec.row_jitter, spec.row_jitter, h)
                   if spec.row_jitter > 0 else np.zeros(h))
        col_off = (rng.uniform(-spec.row_jitter, spec.row_jitter, w)
                   if spec.row_jitter > 0 else np.zeros(w))
        B = max(spec.block_reset, 1)
        if spec.block_reset > 0:
            half = 0.5 * (ow / w)
            bdx = rng.uniform(-half, half, (B, B))
            bdy = rng.uniform(-half, half, (B, B))
        else:
            bdx = bdy = np.zeros((1, 1))
        bx = np.minimum(np.arange(ow) * B // max(ow, 1), B - 1)
        by = np.minimum(np.arange(oh) * B // max(oh, 1), B - 1)
        iy0 = np.clip(np.searchsorted(ycuts, Y, side="right") - 1, 0, h - 1)
        ix0 = np.clip(np.searchsorted(xcuts, X, side="right") - 1, 0, w - 1)
        Sx = row_off[iy0][:, None] + bdx[by[:, None], bx[None, :]]
        Sy = col_off[ix0][None, :] + bdy[by[:, None], bx[None, :]]
        U = _u_of_x(xcuts, np.clip(X[None, :] + Sx, 0, ow)).astype(np.float32)
        V = _u_of_x(ycuts, np.clip(Y[:, None] + Sy, 0, oh)).astype(np.float32)
    else:
        u = _u_of_x(xcuts, X)
        v = _u_of_x(ycuts, Y)
        U = np.broadcast_to(u[None, :], (oh, ow)).astype(np.float32).copy()
        V = np.broadcast_to(v[:, None], (oh, ow)).astype(np.float32).copy()

    ix = np.clip(U.astype(np.int32), 0, w - 1)
    iy = np.clip(V.astype(np.int32), 0, h - 1)
    img = art[iy, ix].astype(np.float32)

    soft = spec.grid_soften
    if soft <= 0 and spec.resample in _PIL_RESAMPLE:
        soft = 0.55  # emulate a smooth resampler over the uneven grid
    if soft > 0:
        img = _gauss(img, soft)
    return img, U, V


def _displace(img: np.ndarray, U: np.ndarray, V: np.ndarray,
              dx: np.ndarray, dy: np.ndarray):
    """Compose a displacement field: out(x,y) = in(x+dx, y+dy).
    U/V are resampled through the same field so they stay exact."""
    oh, ow = img.shape[:2]
    if cv2 is not None:
        yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
        mx = (xx + dx).astype(np.float32)
        my = (yy + dy).astype(np.float32)
        out = cv2.remap(img.astype(np.float32), mx, my,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        U2 = cv2.remap(U, mx, my, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        V2 = cv2.remap(V, mx, my, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        return out, U2, V2
    from scipy.ndimage import map_coordinates
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    coords = [yy + dy, xx + dx]
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = map_coordinates(img[:, :, c], coords, order=1,
                                       mode="nearest")
    U2 = map_coordinates(U.astype(np.float64), coords, order=1,
                         mode="nearest").astype(np.float32)
    V2 = map_coordinates(V.astype(np.float64), coords, order=1,
                         mode="nearest").astype(np.float32)
    return out, U2, V2


def _warp_field(oh, ow, amp, freq, rng):
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    ph = rng.uniform(0, 2 * np.pi, 4)
    dx = amp * np.sin(2 * np.pi * freq * yy / oh + ph[0]) \
        + 0.5 * amp * np.sin(2 * np.pi * freq * 2.3 * yy / oh + ph[1])
    dy = amp * np.sin(2 * np.pi * freq * xx / ow + ph[2]) \
        + 0.5 * amp * np.sin(2 * np.pi * freq * 2.3 * xx / ow + ph[3])
    return dx, dy


def _mush_field(oh, ow, amp, cell_px, rng):
    """Smooth random displacement field. Generated at half resolution and
    upsampled (visually identical for a low-pass field, ~4x cheaper)."""
    from scipy.ndimage import gaussian_filter, zoom
    hh, hw = max(oh // 2, 2), max(ow // 2, 2)
    sigma = max(cell_px * 0.6, 1.0)
    dx = _gauss(rng.standard_normal((hh, hw)).astype(np.float32), sigma)
    dy = _gauss(rng.standard_normal((hh, hw)).astype(np.float32), sigma)
    if cv2 is not None:
        dx = cv2.resize(dx, (ow, oh), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(dy, (ow, oh), interpolation=cv2.INTER_LINEAR)
    else:
        dx = zoom(dx, (oh / hh + 1e-6, ow / hw + 1e-6), order=1)[:oh, :ow]
        dy = zoom(dy, (oh / hh + 1e-6, ow / hw + 1e-6), order=1)[:oh, :ow]
    for d in (dx, dy):
        d *= amp / (np.abs(d).max() + 1e-9)
    return dx, dy


def _smooth_field(oh, ow, rng, cycles=1.5):
    """Low-frequency random field in [-1, 1]. The cubic zoom of a tiny
    grid is already smooth — no full-res gaussian needed (that filter was
    the single hottest op in the data pipeline on large renders)."""
    gh = max(int(cycles * 2) + 2, 3)
    small = rng.standard_normal((gh, gh)).astype(np.float32)
    if cv2 is not None:
        f = cv2.resize(small, (ow, oh), interpolation=cv2.INTER_CUBIC)
    else:
        from scipy.ndimage import zoom
        f = zoom(small, (oh / gh + 1e-6, ow / gh + 1e-6), order=3)[:oh, :ow]
    f -= f.mean()
    f /= (np.abs(f).max() + 1e-9)
    return f


def _jpeg(img_u8: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img_u8, "RGB")
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _webp(img_u8: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img_u8, "RGB")
    buf = io.BytesIO()
    pil.save(buf, "WEBP", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


_CHAIN_FILTERS = [Image.Resampling.NEAREST, Image.Resampling.BILINEAR,
                  Image.Resampling.BICUBIC, Image.Resampling.LANCZOS,
                  Image.Resampling.BOX]


def _resize_with_uv(img: np.ndarray, U: np.ndarray, V: np.ndarray,
                    nw: int, nh: int, resample) -> tuple:
    """Resize image with an arbitrary filter; U/V are coordinate fields so
    plain bilinear resampling of them stays exact for affine resizes."""
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    img2 = np.asarray(pil.resize((nw, nh), resample), dtype=np.float32)
    U2 = np.asarray(Image.fromarray(U, mode="F")
                    .resize((nw, nh), Image.Resampling.BILINEAR))
    V2 = np.asarray(Image.fromarray(V, mode="F")
                    .resize((nw, nh), Image.Resampling.BILINEAR))
    return img2, U2.copy(), V2.copy()


# ------------------------------------------------------------- driver


def distort(art: np.ndarray, spec: DistortSpec,
            rng: Optional[np.random.Generator] = None,
            texture: Optional[np.ndarray] = None) -> dict:
    """art: uint8 (h, w, 3) native pixel art.
    texture: optional float32 zero-mean field, any size (tiled/cropped to
    the output); modulates brightness when spec.cell_texture > 0.

    Returns dict:
      img   uint8 (oh, ow, 3)  distorted image
      U, V  float32 (oh, ow)   exact source coords (native px units)
      gt    the input art (reference)
      spec  the DistortSpec used
    """
    if rng is None:
        rng = np.random.default_rng(spec.seed)
    assert art.ndim == 3 and art.shape[2] == 3, "expected RGB uint8"

    img, U, V = _render_geometry(art, spec, rng)
    oh, ow = img.shape[:2]

    if spec.subpixel_shift > 0:
        # constant fractional offset: geometry tracked exactly via U/V
        ang = rng.uniform(0, 2 * np.pi)
        sdx = np.full((oh, ow), spec.subpixel_shift * np.cos(ang),
                      np.float32)
        sdy = np.full((oh, ow), spec.subpixel_shift * np.sin(ang),
                      np.float32)
        img, U, V = _displace(img, U, V, sdx, sdy)

    if spec.warp_amp > 0:
        dx, dy = _warp_field(oh, ow, spec.warp_amp, spec.warp_freq, rng)
        img, U, V = _displace(img, U, V, dx, dy)

    if spec.mush > 0:
        dx, dy = _mush_field(oh, ow, spec.mush, (spec.sx + spec.sy) / 2, rng)
        img, U, V = _displace(img, U, V, dx, dy)
        # soft focus that goes with the mushy look
        img = 0.6 * img + 0.4 * _gauss(img, 0.6)

    # ---------------- cell texture (uses exact U/V; painterly fakes) ---
    if spec.cell_gradient > 0:
        hn, wn = art.shape[:2]
        gvx = rng.uniform(-1, 1, (hn, wn))
        gvy = rng.uniform(-1, 1, (hn, wn))
        ix = np.clip(U.astype(np.int32), 0, wn - 1)
        iy = np.clip(V.astype(np.int32), 0, hn - 1)
        fu = U - np.floor(U)
        fv = V - np.floor(V)
        shade = ((fu - 0.5) * gvx[iy, ix] + (fv - 0.5) * gvy[iy, ix])
        img = img + (2.0 * spec.cell_gradient) * shade[:, :, None]

    if spec.cell_noise > 0:
        cell_px = (spec.sx + spec.sy) / 2
        t = _gauss(rng.standard_normal((oh, ow)).astype(np.float32),
                   max(cell_px * 0.30, 0.8))
        t /= np.abs(t).max() + 1e-9
        img = img + spec.cell_noise * t[:, :, None]

    if spec.cell_texture > 0:
        if texture is None:
            texture = _gauss(rng.standard_normal((oh, ow)).astype(np.float32),
                             max((spec.sx + spec.sy) / 5, 1.0))
            texture /= texture.std() + 1e-9
        th_, tw_ = texture.shape[:2]
        ty = np.arange(oh) % th_
        tx = np.arange(ow) % tw_
        t = texture[np.ix_(ty, tx)]
        img = img * (1.0 + spec.cell_texture * t[:, :, None])

    # ---------------- compound resize chain (U/V tracked exactly) ------
    if spec.resize_chain > 0:
        for _ in range(spec.resize_chain):
            fx = float(rng.uniform(0.6, 1.35))
            fy = fx if rng.random() < 0.7 else float(rng.uniform(0.6, 1.35))
            nw2 = max(int(round(img.shape[1] * fx)), 16)
            nh2 = max(int(round(img.shape[0] * fy)), 16)
            filt = _CHAIN_FILTERS[int(rng.integers(len(_CHAIN_FILTERS)))]
            img, U, V = _resize_with_uv(img, U, V, nw2, nh2, filt)
        oh, ow = img.shape[:2]

    # ---------------- photometric chain (U/V untouched) ----------------
    if spec.median:
        from scipy.ndimage import median_filter
        img = median_filter(img, size=(3, 3, 1))

    if spec.downup > 0:
        f = spec.downup
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        mid = pil.resize((max(int(ow * f), 2), max(int(oh * f), 2)),
                         Image.Resampling.BILINEAR)
        img = np.asarray(mid.resize((ow, oh), Image.Resampling.BILINEAR),
                         dtype=np.float32)

    if spec.blur > 0:
        img = _gauss(img, spec.blur)

    if spec.sharpen > 0:
        soft = _gauss(img, 1.0)
        img = img + spec.sharpen * (img - soft)

    if spec.noise > 0:
        img = img + rng.normal(0, spec.noise, size=(oh, ow, 1))

    if spec.chroma_noise > 0:
        img = img + rng.normal(0, spec.chroma_noise, size=(oh, ow, 3))

    if spec.color_field > 0:
        gains = np.stack([1.0 + spec.color_field * _smooth_field(oh, ow, rng)
                          for _ in range(3)], axis=2)
        img = img * gains

    img = np.clip(img, 0, 255).astype(np.uint8)

    if spec.banding > 1:
        lv = spec.banding
        img = (np.round(img.astype(np.float32) / 255 * (lv - 1))
               * (255.0 / (lv - 1))).astype(np.uint8)

    if spec.glow > 0:
        f = img.astype(np.float32)
        luma = f.mean(2)
        thresh = np.percentile(luma, 92)
        bright = np.clip(luma - thresh, 0, None)[:, :, None] / 64.0
        halo = _gauss(f * bright, max((spec.sx + spec.sy) * 0.4, 1.5))
        img = np.clip(f + spec.glow * halo, 0, 255).astype(np.uint8)

    if spec.chroma_sub:
        f = img.astype(np.float32)
        y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        cb = f[..., 2] - y
        cr = f[..., 0] - y
        for c in (cb, cr):
            small = c[::2, ::2]
            up = np.repeat(np.repeat(small, 2, 0), 2, 1)[:c.shape[0],
                                                         :c.shape[1]]
            c[:] = _gauss(up, 0.7)
        r = y + cr
        b = y + cb
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        img = np.clip(np.stack([r, g, b], 2), 0, 255).astype(np.uint8)

    if spec.jpeg_quality:
        img = _jpeg(img, spec.jpeg_quality)
        if spec.jpeg_twice:
            img = _jpeg(img, min(spec.jpeg_quality + 15, 95))

    if spec.webp_quality:
        img = _webp(img, spec.webp_quality)

    return {"img": img, "U": U, "V": V, "gt": art, "spec": spec}


# ------------------------------------------------ random spec sampling


def sample_spec(rng: np.random.Generator, difficulty: str = "mixed",
                min_scale: float = 1.25, max_scale: float = 16.0) -> DistortSpec:
    """Random distortion pipeline. difficulty: easy|medium|hard|extreme|mixed.

    'mixed' draws a tier per call (weighted toward hard, since the wild is
    ugly). Scales are log-uniform and include small upscales; non-square
    scale pairs appear ~35% of the time.
    """
    if difficulty == "mixed":
        difficulty = rng.choice(["easy", "medium", "hard", "extreme"],
                                p=[0.15, 0.30, 0.35, 0.20])
    # real-world-shaped scale law: small upscales dominate (1.25-4x),
    # then 4-8x, then 8-16x; log-uniform within each bucket
    b = rng.random()
    if b < 0.50:
        blo, bhi = min_scale, min(4.0, max_scale)
    elif b < 0.82:
        blo, bhi = min(4.0, max_scale), min(8.0, max_scale)
    else:
        blo, bhi = min(8.0, max_scale), max_scale
    sx = float(np.exp(rng.uniform(np.log(blo), np.log(max(bhi, blo + 1e-6)))))
    sy = sx
    r = rng.random()
    if r < 0.20:          # mildly non-square
        sy = float(np.clip(sx * rng.uniform(0.75, 1.35), min_scale, max_scale))
    elif r < 0.35:        # strongly non-square (independent axes, same
        sy = float(np.clip(  # bucket so aspect stays plausible)
            np.exp(rng.uniform(np.log(blo), np.log(max(bhi, blo + 1e-6)))),
            min_scale, max_scale))

    def u(a, b):
        return float(rng.uniform(a, b))

    spec = DistortSpec(sx=sx, sy=sy, seed=int(rng.integers(1 << 31)))
    spec.resample = str(rng.choice(["nearest", "bilinear", "bicubic"],
                                   p=[0.45, 0.35, 0.20]))

    # NOTE: per-cut jitter ("grid shuffle") is deliberately absent — real
    # inputs (AI output, bad resizes) never have randomly offset cuts; the
    # real failure modes are smooth drift, row/block phase offsets and mush.
    if difficulty == "easy":
        if rng.random() < 0.08:
            spec.subpixel_shift = u(0.25, 0.6)
        if rng.random() < 0.3:
            spec.jpeg_quality = int(rng.integers(75, 96))
        if rng.random() < 0.3:
            spec.blur = u(0.2, 0.5)
        if rng.random() < 0.2:
            spec.noise = u(1, 4)
    elif difficulty == "medium":
        if False:  # chroma_sub removed (run-8 postmortem)
            spec.chroma_sub = True
        if rng.random() < 0.08:
            spec.glow = u(0.3, 0.8)
        if rng.random() < 0.12:
            spec.subpixel_shift = u(0.25, 0.7)
        if rng.random() < 0.30:
            spec.resize_chain = 1
        if rng.random() < 0.15:
            spec.webp_quality = int(rng.integers(60, 91))
        if rng.random() < 0.08:
            spec.cell_noise = u(3, 8)
        if rng.random() < 0.35:
            spec.drift = u(0.03, 0.10)
        if rng.random() < 0.4:
            spec.jpeg_quality = int(rng.integers(60, 90))
        if rng.random() < 0.4:
            spec.blur = u(0.3, 0.8)
        if rng.random() < 0.3:
            spec.noise = u(2, 6)
        if rng.random() < 0.25:
            spec.chroma_noise = u(2, 6)
        if rng.random() < 0.2:
            spec.downup = u(0.75, 0.95)
        if rng.random() < 0.15:
            spec.warp_amp, spec.warp_freq = u(0.5, 1.5), u(1.5, 3.0)
    elif difficulty == "hard":
        # painterly is EXCLUSIVE and capped (independent rolls stacked to
        # >50% of the corpus once — awful balance)
        r_p = rng.random()
        if r_p < 0.10:
            spec.cell_gradient = u(8, 25)
        elif r_p < 0.18:
            spec.cell_noise = u(4, 12)
        elif r_p < 0.26:
            spec.cell_texture = u(0.08, 0.22)
        if False:  # chroma_sub removed (run-8 postmortem)
            spec.chroma_sub = True
        if rng.random() < 0.12:
            spec.glow = u(0.4, 1.2)
        if rng.random() < 0.15:
            spec.subpixel_shift = u(0.25, 0.7)
        if rng.random() < 0.45:
            spec.resize_chain = int(rng.integers(1, 3))
        if rng.random() < 0.20:
            spec.webp_quality = int(rng.integers(50, 86))
        if rng.random() < 0.10:
            spec.median = True
        if rng.random() < 0.45:
            spec.drift = u(0.06, 0.15)
        if rng.random() < 0.35:
            spec.row_jitter = u(0.3, 1.2)
        if rng.random() < 0.25:
            spec.block_reset = int(rng.integers(2, 5))
        if rng.random() < 0.45:
            spec.mush = u(0.6, 1.6)
        if rng.random() < 0.5:
            spec.blur = u(0.5, 1.2)
        if rng.random() < 0.3:
            spec.sharpen = u(0.3, 1.0)
        if rng.random() < 0.5:
            spec.jpeg_quality = int(rng.integers(45, 80))
        if rng.random() < 0.4:
            spec.noise = u(3, 9)
        if rng.random() < 0.35:
            spec.chroma_noise = u(3, 9)
        if rng.random() < 0.3:
            spec.color_field = u(0.05, 0.15)
        if rng.random() < 0.25:
            spec.downup = u(0.6, 0.9)
        if rng.random() < 0.2:
            spec.warp_amp, spec.warp_freq = u(1.0, 2.5), u(1.5, 3.5)
    else:  # extreme
        r_p = rng.random()
        if r_p < 0.12:
            spec.cell_gradient = u(12, 40)
        elif r_p < 0.22:
            spec.cell_noise = u(6, 16)
        elif r_p < 0.32:
            spec.cell_texture = u(0.12, 0.30)
        if False:  # chroma_sub removed (run-8 postmortem)
            spec.chroma_sub = True
        if rng.random() < 0.15:
            spec.glow = u(0.5, 1.5)
        if rng.random() < 0.15:
            spec.subpixel_shift = u(0.3, 0.7)
        if rng.random() < 0.60:
            spec.resize_chain = int(rng.integers(1, 3))
        if rng.random() < 0.25:
            spec.webp_quality = int(rng.integers(40, 81))
        if rng.random() < 0.15:
            spec.median = True
        if rng.random() < 0.6:
            spec.drift = u(0.10, 0.20)
        if rng.random() < 0.5:
            spec.row_jitter = u(0.8, 2.0)
        if rng.random() < 0.35:
            spec.block_reset = int(rng.integers(2, 6))
        if rng.random() < 0.7:
            spec.mush = u(1.2, 2.8)
        if rng.random() < 0.7:
            spec.blur = u(0.8, 1.6)
        if rng.random() < 0.4:
            spec.sharpen = u(0.5, 1.5)
        if rng.random() < 0.6:
            spec.jpeg_quality = int(rng.integers(35, 70))
            spec.jpeg_twice = rng.random() < 0.4
        if rng.random() < 0.5:
            spec.noise = u(5, 14)
        if rng.random() < 0.5:
            spec.chroma_noise = u(5, 14)
        if rng.random() < 0.4:
            spec.color_field = u(0.10, 0.25)
        if rng.random() < 0.3:
            spec.banding = int(rng.integers(8, 32))
        if rng.random() < 0.35:
            spec.downup = u(0.55, 0.85)
        if rng.random() < 0.35:
            spec.warp_amp, spec.warp_freq = u(1.5, 3.0), u(2.0, 4.0)

    # painterly cell texture implies BIG-ish pseudo-cells (sub-3px cells
    # can't express internal texture); lift only when needed and keep the
    # 4-6x painterly regime represented too
    if spec.cell_gradient > 0 or spec.cell_noise > 0 or spec.cell_texture > 0:
        m = (spec.sx + spec.sy) / 2
        if m < 4.0:
            f = float(rng.uniform(4.0, 12.0)) / m
            spec.sx = min(spec.sx * f, max_scale)
            spec.sy = min(spec.sy * f, max_scale)

    spec.name = f"{difficulty}_x{spec.sx:.2f}x{spec.sy:.2f}"
    return spec


# Named extreme specs for a reproducible static suite (gallery + eval).
EXTREME_SPECS = [
    DistortSpec(name="tiny_x1p5", sx=1.5, sy=1.5, seed=101),
    DistortSpec(name="tiny_x2p3_bilinear", sx=2.3, sy=2.3, resample="bilinear",
                seed=102),
    DistortSpec(name="small_x2p8_jpeg", sx=2.8, sy=2.8, jpeg_quality=65,
                seed=103),
    DistortSpec(name="nonsquare_x4x6", sx=4.0, sy=6.0, seed=104),
    DistortSpec(name="nonsquare_x3p2x5p7_soft", sx=3.2, sy=5.7,
                resample="bilinear", blur=0.5, seed=105),
    DistortSpec(name="frac_x4p37", sx=4.37, sy=4.37, resample="bicubic",
                seed=106),
    DistortSpec(name="drift_heavy_x5", sx=5.0, sy=5.0, drift=0.18,
                seed=107),
    DistortSpec(name="drift_mush_x4p3", sx=4.3, sy=4.3, drift=0.12,
                mush=1.2, resample="bilinear", seed=108),
    DistortSpec(name="blur_heavy_x4p6", sx=4.6, sy=4.6, resample="bilinear",
                blur=1.5, seed=109),
    DistortSpec(name="mush_extreme_x5p2", sx=5.2, sy=5.2, resample="bilinear",
                mush=2.6, blur=0.9, seed=110),
    DistortSpec(name="noise_heavy_x4", sx=4.0, sy=4.0, noise=12.0, blur=0.4,
                seed=111),
    DistortSpec(name="chroma_noise_x4p8", sx=4.8, sy=4.8, chroma_noise=12.0,
                color_field=0.15, seed=112),
    DistortSpec(name="jpeg40_twice_x5", sx=5.0, sy=5.0, resample="bilinear",
                jpeg_quality=40, jpeg_twice=True, seed=113),
    DistortSpec(name="downup_x3p3", sx=3.3, sy=3.3, downup=0.62, seed=114),
    DistortSpec(name="warp_mush_x5p5", sx=5.5, sy=5.5, warp_amp=2.5,
                warp_freq=3.0, mush=1.5, seed=115),
    DistortSpec(name="rowjit_blocks_x5", sx=5.0, sy=5.0, row_jitter=1.5,
                block_reset=4, mush=0.8, seed=116),
    DistortSpec(name="sharpen_halo_x4p1", sx=4.1, sy=4.1, resample="bilinear",
                blur=0.8, sharpen=1.4, jpeg_quality=75, seed=117),
    DistortSpec(name="banding_x4p4", sx=4.4, sy=4.4, resample="bilinear",
                banding=12, chroma_noise=4.0, seed=118),
    DistortSpec(name="kitchen_extreme_x5p6", sx=5.6, sy=5.6,
                resample="bilinear", drift=0.12, row_jitter=1.0,
                mush=1.8, blur=1.0, sharpen=0.6, noise=6.0, chroma_noise=6.0,
                color_field=0.12, jpeg_quality=55, downup=0.8, seed=119),
    DistortSpec(name="kitchen_small_x2p4", sx=2.4, sy=2.4,
                resample="bilinear", mush=0.8, blur=0.5,
                noise=4.0, chroma_noise=4.0, jpeg_quality=70, seed=120),
    # painterly AI fakes: big pseudo-cells with internal texture
    DistortSpec(name="painterly_x8", sx=8.0, sy=8.0, cell_gradient=25.0,
                cell_noise=10.0, mush=1.0, blur=0.6, seed=121),
    DistortSpec(name="painterly_x9p5_jpeg", sx=9.5, sy=9.5,
                cell_gradient=32.0, cell_noise=12.0, drift=0.06,
                blur=0.8, jpeg_quality=70, seed=122),
]


# --------------------------------------------------- target computation


def targets_from_uv(U: np.ndarray, V: np.ndarray):
    """Dense training targets from exact source-coordinate maps.

    Returns dict of float32 (oh, ow):
      sinu, cosu, sinv, cosv : phase embedding of the grid (period = 1 cell)
      bnd_v, bnd_h           : gaussian heatmaps of vertical/horizontal cuts
                               (distance measured in OUTPUT pixels)
      logfx, logfy           : log2 local cell size in output px
    """
    out = {}
    tau = 2 * np.pi
    out["sinu"] = np.sin(tau * U).astype(np.float32)
    out["cosu"] = np.cos(tau * U).astype(np.float32)
    out["sinv"] = np.sin(tau * V).astype(np.float32)
    out["cosv"] = np.cos(tau * V).astype(np.float32)

    # local frequency (cells per output px) from finite differences
    def freq(M, axis):
        g = np.gradient(M, axis=axis)
        return np.clip(np.abs(g), 1e-4, 4.0)

    fx = freq(U, 1)
    fy = freq(V, 0)
    out["logfx"] = np.log2(1.0 / fx).astype(np.float32)   # log2 cell size px
    out["logfy"] = np.log2(1.0 / fy).astype(np.float32)

    # boundary heatmap: distance to nearest integer of U, in output px.
    # sigma scales with the local cell size so small cells keep distinct
    # peaks instead of blurring into an all-boundary target.
    def heat(M, f):
        frac = M - np.round(M)             # in (-0.5, 0.5], 0 at a cut
        d_px = np.abs(frac) / f            # convert cells -> output px
        sigma = np.clip(0.22 / f, 0.30, 1.0)   # 1/f = cell size in px
        return np.exp(-0.5 * (d_px / sigma) ** 2).astype(np.float32)

    out["bnd_v"] = heat(U, fx)
    out["bnd_h"] = heat(V, fy)
    return out


def spec_dict(spec: DistortSpec) -> dict:
    return asdict(spec)
