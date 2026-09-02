from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelbench.metrics.expanded import score_reconstruction


OUTPUT = ROOT / "validation"
BG, PANEL, TEXT, MUTED, GRID, GOLD, GREEN = (
    (7, 7, 6), (16, 16, 13), (245, 240, 225), (163, 157, 140),
    (53, 51, 43), (218, 184, 94), (74, 190, 126),
)


def font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def sprite() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[3:13, 3:13] = (150, 35, 170)
    image[5:11, 5:11] = (35, 40, 55)
    image[7, 7] = (45, 230, 90)
    return image


def monotonic(values: list[float], increasing: bool = True, tolerance: float = 1e-9) -> bool:
    differences = np.diff(np.asarray(values, dtype=np.float64))
    ordered = bool(np.all(differences >= -tolerance)) if increasing else bool(np.all(differences <= tolerance))
    changed = values[-1] > values[0] + tolerance if increasing else values[-1] < values[0] - tolerance
    return ordered and changed


def validation_curves() -> list[dict]:
    target = sprite()

    drift_x = [0, 10, 20, 30, 40]
    drift_scores = []
    for amount in drift_x:
        changed = target.copy().astype(np.int16)
        changed[3:13, 3:13, 0] += amount
        drift_scores.append(score_reconstruction(target, np.clip(changed, 0, 255).astype(np.uint8)))

    shift_x = [0, 1, 2, 3]
    shift_scores = []
    for shift in shift_x:
        changed = np.zeros_like(target)
        if shift == 0:
            changed[:] = target
        else:
            changed[:, shift:] = target[:, :-shift]
        shift_scores.append(score_reconstruction(target, changed))

    blend_x = [0, 20, 40, 60, 80, 100]
    blend_scores = []
    for percent in blend_x:
        count = round(10 * percent / 100)
        changed = target.copy()
        changed[3, 3:3 + count] = (
            (target[3, 3:3 + count].astype(np.uint16) + target[2, 3:3 + count].astype(np.uint16)) // 2
        ).astype(np.uint8)
        blend_scores.append(score_reconstruction(target, changed))

    palette_colours = np.asarray(
        [(20, 20, 30), (80, 30, 120), (180, 60, 150), (40, 220, 100)],
        dtype=np.uint8,
    )
    palette_target = np.zeros((16, 16, 3), dtype=np.uint8)
    for index, colour in enumerate(palette_colours):
        palette_target[:, index * 4:(index + 1) * 4] = colour
    delete_x = [0, 1, 2, 3]
    delete_scores = []
    for deleted in delete_x:
        changed = palette_target.copy()
        for index in range(4 - deleted, 4):
            changed[:, index * 4:(index + 1) * 4] = palette_colours[0]
        delete_scores.append(score_reconstruction(palette_target, changed))

    curves = [
        {"name": "Mean perceptual colour error", "x_label": "Red-channel drift", "metric": "delta_e00_mean", "x": drift_x, "y": [s["delta_e00_mean"] for s in drift_scores]},
        {"name": "Tail perceptual colour error", "x_label": "Red-channel drift", "metric": "delta_e00_p95", "x": drift_x, "y": [s["delta_e00_p95"] for s in drift_scores]},
        {"name": "Symmetric edge distance", "x_label": "Translation (native pixels)", "metric": "edge_assd_px", "x": shift_x, "y": [s["edge_assd_px"] for s in shift_scores]},
        {"name": "Local edge-palette violations", "x_label": "Blended edge coverage (%)", "metric": "edge_palette_violation_rate", "x": blend_x, "y": [100.0 * s["edge_palette_violation_rate"] for s in blend_scores], "suffix": "%"},
        {"name": "Weighted palette transport", "x_label": "Deleted palette colours", "metric": "palette_emd_de00", "x": delete_x, "y": [s["palette_emd_de00"] for s in delete_scores]},
        {"name": "Symmetric palette distance", "x_label": "Deleted palette colours", "metric": "palette_chamfer_de00", "x": delete_x, "y": [s["palette_chamfer_de00"] for s in delete_scores]},
    ]
    for curve in curves:
        curve["monotonic_pass"] = monotonic(curve["y"])
    return curves


def draw_chart(curves: list[dict], path: Path) -> None:
    image = Image.new("RGB", (1600, 1280), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((34, 25, 46, 84), radius=6, fill=GOLD)
    draw.text((64, 25), "Metric behaviour validation", font=font(32, True), fill=TEXT)
    draw.text((64, 66), "Controlled perturbations must produce monotonic, correctly directed responses", font=font(15), fill=MUTED)
    for index, curve in enumerate(curves):
        col, row = index % 2, index // 2
        x0, y0 = 34 + col * 782, 108 + row * 372
        x1, y1 = x0 + 750, y0 + 342
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=PANEL)
        draw.text((x0 + 18, y0 + 14), curve["name"], font=font(17, True), fill=TEXT)
        status = "PASS" if curve["monotonic_pass"] else "FAIL"
        draw.text((x1 - 18, y0 + 17), status, font=font(13, True), fill=GREEN if status == "PASS" else (194, 72, 69), anchor="ra")
        left, top, right, bottom = x0 + 72, y0 + 62, x1 - 25, y1 - 55
        draw.rectangle((left, top, right, bottom), outline=GRID, width=1)
        values = np.asarray(curve["y"], dtype=float)
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.12, 0.05)
        low, high = min(0.0, low - padding), high + padding
        points = []
        for point_index, value in enumerate(values):
            x = left + (right - left) * point_index / max(len(values) - 1, 1)
            y = bottom - (bottom - top) * (float(value) - low) / max(high - low, 1e-9)
            points.append((x, y))
        draw.line(points, fill=GOLD, width=4)
        suffix = curve.get("suffix", "")
        for point_index, ((x, y), value, x_value) in enumerate(zip(points, values, curve["x"])):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=GREEN)
            anchor = "la" if point_index == 0 else ("ra" if point_index == len(points) - 1 else "ma")
            draw.text((x, y - 23), f"{value:.2f}{suffix}", font=font(12, True), fill=TEXT, anchor=anchor)
            draw.text((x, bottom + 14), str(x_value), font=font(12), fill=MUTED, anchor="ma")
        draw.text(((left + right) / 2, y1 - 22), curve["x_label"], font=font(12), fill=MUTED, anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    curves = validation_curves()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "bm-metric-validation-v1",
        "rule": "Each retained primary metric must respond monotonically and in the documented direction to its targeted controlled perturbation.",
        "all_pass": all(curve["monotonic_pass"] for curve in curves),
        "curves": curves,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Metric behaviour validation", "",
        "Controlled perturbations isolate one failure mode at a time. Passing means the metric responds monotonically in the documented direction; it does not establish universal perceptual validity.", "",
        "| Metric | Perturbation | Values | Result |", "|---|---|---|---|",
    ]
    for curve in curves:
        values = ", ".join(f"{value:.4f}" for value in curve["y"])
        lines.append(f"| `{curve['metric']}` | {curve['x_label']} | {values} | {'PASS' if curve['monotonic_pass'] else 'FAIL'} |")
    (OUTPUT / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw_chart(curves, OUTPUT / "metric-behaviour.png")
    if not payload["all_pass"]:
        raise SystemExit("One or more metric behaviour checks failed")
    print(f"PASS: {len(curves)} controlled metric curves -> {OUTPUT}")


if __name__ == "__main__":
    main()
