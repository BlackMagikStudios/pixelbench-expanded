from __future__ import annotations

import unittest

import numpy as np

from pixelbench.metrics.expanded import score_reconstruction


def reference_sprite() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[3:13, 3:13] = (150, 35, 170)
    image[5:11, 5:11] = (35, 40, 55)
    image[7, 7] = (45, 230, 90)
    return image


class MetricBehaviourTests(unittest.TestCase):
    def test_identity_is_ideal(self) -> None:
        target = reference_sprite()
        score = score_reconstruction(target, target)
        self.assertAlmostEqual(score["delta_e00_mean"], 0.0, places=6)
        self.assertAlmostEqual(score["delta_e00_p95"], 0.0, places=6)
        self.assertAlmostEqual(score["palette_emd_de00"], 0.0, places=6)
        self.assertAlmostEqual(score["palette_chamfer_de00"], 0.0, places=6)
        self.assertAlmostEqual(score["edge_palette_violation_rate"], 0.0, places=6)
        self.assertAlmostEqual(score["edge_assd_px"], 0.0, places=6)
        self.assertAlmostEqual(score["edge_hd95_px"], 0.0, places=6)
        self.assertAlmostEqual(score["edge_exact_f1"], 1.0, places=6)
        self.assertAlmostEqual(score["ssim"], 1.0, places=6)

    def test_wrong_colour_increases_colour_and_palette_error(self) -> None:
        target = reference_sprite()
        wrong = target.copy()
        wrong[3:13, 3:13] = np.clip(wrong[3:13, 3:13].astype(int) + (45, -20, -60), 0, 255)
        score = score_reconstruction(target, wrong.astype(np.uint8))
        self.assertGreater(score["delta_e00_mean"], 1.0)
        self.assertGreater(score["palette_emd_de00"], 1.0)

    def test_blended_edge_is_counted_as_a_palette_violation(self) -> None:
        target = reference_sprite()
        blended = target.copy()
        blended[3, 3:13] = ((target[3, 3:13].astype(np.uint16) + target[2, 3:13]) // 2).astype(np.uint8)
        score = score_reconstruction(target, blended)
        self.assertGreater(score["edge_palette_violation_rate"], 0.0)

    def test_tolerant_edge_score_exceeds_exact_for_one_pixel_shift(self) -> None:
        target = reference_sprite()
        shifted = np.zeros_like(target)
        shifted[:, 1:] = target[:, :-1]
        score = score_reconstruction(target, shifted)
        self.assertGreater(score["edge_tolerant_f1"], score["edge_exact_f1"])
        self.assertGreater(score["edge_assd_px"], 0.0)
        self.assertGreater(score["edge_hd95_px"], 0.0)

    def test_invented_colour_increases_symmetric_palette_distance(self) -> None:
        target = reference_sprite()
        sprayed = target.copy()
        sprayed[::2, ::2] = (20, 210, 240)
        score = score_reconstruction(target, sprayed)
        self.assertGreater(score["palette_chamfer_de00"], 0.0)
        self.assertGreater(score["palette_emd_de00"], 0.0)

    def test_deleted_accent_increases_symmetric_palette_distance(self) -> None:
        target = reference_sprite()
        deleted = target.copy()
        deleted[7, 7] = deleted[7, 8]
        score = score_reconstruction(target, deleted)
        self.assertGreater(score["palette_chamfer_de00"], 0.0)

    def test_colour_error_increases_monotonically_with_drift(self) -> None:
        target = reference_sprite()
        values = []
        for amount in (0, 10, 20, 30, 40):
            changed = target.copy().astype(np.int16)
            changed[3:13, 3:13, 0] += amount
            changed = np.clip(changed, 0, 255).astype(np.uint8)
            values.append(score_reconstruction(target, changed)["delta_e00_mean"])
        self.assertTrue(all(right >= left for left, right in zip(values, values[1:])))
        self.assertGreater(values[-1], values[0])

    def test_edge_distance_increases_with_translation(self) -> None:
        target = reference_sprite()
        values = []
        for shift in range(4):
            changed = np.zeros_like(target)
            if shift == 0:
                changed[:] = target
            else:
                changed[:, shift:] = target[:, :-shift]
            values.append(score_reconstruction(target, changed)["edge_assd_px"])
        self.assertTrue(all(right >= left for left, right in zip(values, values[1:])))
        self.assertGreater(values[-1], values[0])

    def test_edge_palette_threshold_has_expected_direction(self) -> None:
        target = reference_sprite()
        blended = target.copy()
        blended[3, 3:13] = ((target[3, 3:13].astype(np.uint16) + target[2, 3:13]) // 2).astype(np.uint8)
        values = [
            score_reconstruction(target, blended, de00_threshold=threshold)["edge_palette_violation_rate"]
            for threshold in (1.0, 2.0, 3.0)
        ]
        self.assertTrue(all(right <= left for left, right in zip(values, values[1:])))

    def test_valid_one_pixel_edge_colour_movement_is_not_a_palette_violation(self) -> None:
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        target[:, 8:] = (180, 50, 140)
        shifted = np.zeros_like(target)
        shifted[:, 9:] = (180, 50, 140)
        score = score_reconstruction(target, shifted)
        self.assertEqual(score["edge_palette_violation_rate"], 0.0)
        self.assertGreater(score["edge_assd_px"], 0.0)

    def test_interior_colour_error_is_not_mislabeled_as_edge_palette_damage(self) -> None:
        target = np.zeros((24, 24, 3), dtype=np.uint8)
        target[3:21, 3:21] = (150, 35, 170)
        changed = target.copy()
        changed[10:14, 10:14] = (80, 80, 95)
        score = score_reconstruction(target, changed)
        self.assertGreater(score["delta_e00_mean"], 0.0)
        self.assertEqual(score["edge_palette_violation_rate"], 0.0)

    def test_colour_drift_without_shape_change_preserves_edge_distance(self) -> None:
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        target[4:12, 4:12] = (130, 40, 150)
        changed = np.zeros_like(target)
        changed[4:12, 4:12] = (170, 55, 125)
        score = score_reconstruction(target, changed)
        self.assertGreater(score["delta_e00_mean"], 0.0)
        self.assertEqual(score["edge_assd_px"], 0.0)

    def test_identical_dither_pattern_is_not_called_damage(self) -> None:
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        target[::2, ::2] = (220, 210, 170)
        target[1::2, 1::2] = (220, 210, 170)
        score = score_reconstruction(target, target)
        self.assertEqual(score["edge_palette_violation_rate"], 0.0)
        self.assertEqual(score["palette_chamfer_de00"], 0.0)


if __name__ == "__main__":
    unittest.main()
