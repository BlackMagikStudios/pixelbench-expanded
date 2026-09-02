from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from .base import Metric, register


DE00_THRESHOLD = 2.0
EDGE_THRESHOLD = 2.0
MAX_PALETTE = 256


def _align(native: np.ndarray, target: np.ndarray) -> np.ndarray:
    native = np.asarray(native, dtype=np.uint8)[..., :3]
    height, width = target.shape[:2]
    if native.shape[:2] == (height, width):
        return np.ascontiguousarray(native)
    return np.asarray(
        Image.fromarray(native).resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, dtype=np.float32) / 255.0).astype(np.float32)


def _de00(left_lab: np.ndarray, right_lab: np.ndarray) -> np.ndarray:
    return np.asarray(deltaE_ciede2000(left_lab, right_lab), dtype=np.float32)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    size = 2 * radius + 1
    return binary_dilation(mask, structure=np.ones((size, size), dtype=bool))


def _edge_mask(lab: np.ndarray, threshold: float = EDGE_THRESHOLD) -> np.ndarray:
    edge = np.zeros(lab.shape[:2], dtype=bool)
    horizontal = _de00(lab[:, 1:], lab[:, :-1]) > threshold
    vertical = _de00(lab[1:], lab[:-1]) > threshold
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    return edge


def _mask_f1(reference: np.ndarray, candidate: np.ndarray, tolerance: int = 0) -> float:
    reference = reference.astype(bool)
    candidate = candidate.astype(bool)
    if not reference.any() and not candidate.any():
        return 1.0
    reference_near = _dilate(reference, tolerance)
    candidate_near = _dilate(candidate, tolerance)
    precision = float((candidate & reference_near).sum()) / max(int(candidate.sum()), 1)
    recall = float((reference & candidate_near).sum()) / max(int(reference.sum()), 1)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _local_min_de(reference_lab: np.ndarray, candidate_lab: np.ndarray) -> np.ndarray:
    height, width = reference_lab.shape[:2]
    padded = np.pad(candidate_lab, ((1, 1), (1, 1), (0, 0)), mode="edge")
    best = np.full((height, width), np.inf, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            shifted = padded[dy:dy + height, dx:dx + width]
            np.minimum(best, _de00(reference_lab, shifted), out=best)
    return best


def _edge_surface_distances(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """ASSD and robust Hausdorff (95th percentile) in native pixel units."""
    reference = reference.astype(bool)
    candidate = candidate.astype(bool)
    if not reference.any() and not candidate.any():
        return 0.0, 0.0
    if not reference.any() or not candidate.any():
        penalty = math.hypot(*reference.shape)
        return float(penalty), float(penalty)
    distance_to_candidate = distance_transform_edt(~candidate)
    distance_to_reference = distance_transform_edt(~reference)
    distances = np.concatenate((
        distance_to_candidate[reference],
        distance_to_reference[candidate],
    ))
    return float(distances.mean()), float(np.percentile(distances, 95))


def _palette(rgb: np.ndarray, maximum: int = MAX_PALETTE) -> tuple[np.ndarray, np.ndarray, int]:
    flat = np.asarray(rgb, dtype=np.uint8)[..., :3].reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    exact_count = len(colours)
    if exact_count > maximum:
        quantized = Image.fromarray(rgb).quantize(
            colors=maximum,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        ).convert("RGB")
        colours, counts = np.unique(np.asarray(quantized).reshape(-1, 3), axis=0, return_counts=True)
    weights = counts.astype(np.float32)
    weights /= max(float(weights.sum()), 1.0)
    return colours.astype(np.uint8), weights, exact_count


def _palette_distances(target: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    target_colours, target_weights, target_count = _palette(target)
    candidate_colours, candidate_weights, candidate_count = _palette(candidate)
    target_lab = _lab(target_colours.reshape(1, -1, 3)).reshape(-1, 3)
    candidate_lab = _lab(candidate_colours.reshape(1, -1, 3)).reshape(-1, 3)
    cost = _de00(target_lab[:, None, :], candidate_lab[None, :, :])
    target_signature = np.column_stack((target_weights, target_lab)).astype(np.float32)
    candidate_signature = np.column_stack((candidate_weights, candidate_lab)).astype(np.float32)
    emd, _, _ = cv2.EMD(
        target_signature,
        candidate_signature,
        cv2.DIST_USER,
        cost.astype(np.float32),
    )
    target_to_candidate = cost.min(axis=1)
    candidate_to_target = cost.min(axis=0)
    symmetric_chamfer = 0.5 * (
        float(target_to_candidate.mean()) + float(candidate_to_target.mean())
    )
    size_error = abs(math.log2((candidate_count + 1.0) / (target_count + 1.0)))
    return float(emd), float(symmetric_chamfer), float(size_error)


def score_reconstruction(
    target: np.ndarray,
    native: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    *,
    de00_threshold: float = DE00_THRESHOLD,
    edge_threshold: float = EDGE_THRESHOLD,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.uint8)[..., :3]
    size_exact = native.shape[:2] == target.shape[:2]
    candidate = _align(native, target)
    target_lab = _lab(target)
    candidate_lab = _lab(candidate)
    aligned_de = _de00(candidate_lab, target_lab)
    local_de = _local_min_de(candidate_lab, target_lab)

    diff = candidate.astype(np.float64) - target.astype(np.float64)
    mse = float(np.square(diff).mean())
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0 ** 2) / mse)
    min_side = min(target.shape[:2])
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    ssim = structural_similarity(
        target,
        candidate,
        channel_axis=2,
        data_range=255,
        win_size=max(3, win_size),
    )

    target_edge = _edge_mask(target_lab, edge_threshold)
    candidate_edge = _edge_mask(candidate_lab, edge_threshold)
    edge_band = _dilate(target_edge, 1)
    edge_palette_violation = (
        float((local_de[edge_band] > de00_threshold).mean()) if edge_band.any() else 0.0
    )
    edge_assd, edge_hd95 = _edge_surface_distances(target_edge, candidate_edge)

    palette_emd, palette_chamfer, palette_size_error = _palette_distances(target, candidate)

    if foreground_mask is None or foreground_mask.shape != target.shape[:2] or not foreground_mask.any():
        foreground_mask = np.ones(target.shape[:2], dtype=bool)

    scores = {
        "delta_e00_mean": float(aligned_de.mean()),
        "delta_e00_p95": float(np.percentile(aligned_de, 95)),
        "foreground_delta_e00_mean": float(aligned_de[foreground_mask].mean()),
        "local_delta_e00_mean": float(local_de.mean()),
        "color_match_de00_2": float((aligned_de <= de00_threshold).mean()),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "pixel_match_tol4": float(
            (np.abs(candidate.astype(np.int16) - target.astype(np.int16)).max(axis=2) <= 4).mean()
        ),
        "edge_exact_f1": _mask_f1(target_edge, candidate_edge, tolerance=0),
        "edge_tolerant_f1": _mask_f1(target_edge, candidate_edge, tolerance=1),
        "edge_assd_px": edge_assd,
        "edge_hd95_px": edge_hd95,
        "edge_palette_violation_rate": edge_palette_violation,
        "palette_emd_de00": palette_emd,
        "palette_chamfer_de00": palette_chamfer,
        "palette_size_log_error": palette_size_error,
        "native_size_exact": float(size_exact),
    }
    return scores


@register
class ExpandedQuality(Metric):
    """Complementary full-reference quality signals for native pixel art."""

    name = "expanded_quality"
    fields = {
        "delta_e00_mean": "lower",
        "delta_e00_p95": "lower",
        "foreground_delta_e00_mean": "lower",
        "local_delta_e00_mean": "lower",
        "color_match_de00_2": "higher",
        "ssim": "higher",
        "pixel_match_tol4": "higher",
        "edge_exact_f1": "higher",
        "edge_tolerant_f1": "higher",
        "edge_assd_px": "lower",
        "edge_hd95_px": "lower",
        "edge_palette_violation_rate": "lower",
        "palette_emd_de00": "lower",
        "palette_chamfer_de00": "lower",
        "palette_size_log_error": "lower",
        "native_size_exact": "higher",
    }

    def score(self, sample, rec) -> dict[str, float]:
        if rec.native is None:
            return {}
        scores = score_reconstruction(sample.gt, rec.native)
        # The helper also computes upstream-compatible PSNR for standalone
        # analysis. Do not register that duplicate field here: PixelBench's
        # unchanged ``color`` metric remains the sole owner of ``psnr``.
        return {field: scores[field] for field in self.fields}


__all__ = ["ExpandedQuality", "score_reconstruction"]
