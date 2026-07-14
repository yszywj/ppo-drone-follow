from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pegasus_iris_fast_line_follow.motion_task import MotionPoolConfig, MotionTaskGenerator


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class MotionTaskGeneratorTest(unittest.TestCase):
    def _load_pool(self, name: str) -> MotionPoolConfig:
        data = json.loads((PACKAGE_DIR / "configs" / name).read_text(encoding="utf-8"))
        return MotionPoolConfig.from_dict(data["task"]["motion_pool"])

    def test_speed_pool_respects_limits_and_stops(self) -> None:
        config = self._load_pool("stage1_speed_pool_5hz_ratio_5to5.json")
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(43)
        for _ in range(32):
            trajectory = generator.sample(rng, [0.0, 0.0, -5.0], rng.uniform(-np.pi, np.pi))
            horizontal_speed = np.linalg.norm(trajectory.velocities[:, :2], axis=1)
            self.assertLessEqual(float(horizontal_speed.max()), config.limits.max_speed_mps + 1e-6)
            self.assertLessEqual(
                float(np.abs(trajectory.velocities[:, 2]).max()),
                config.limits.max_vertical_speed_mps + 1e-6,
            )
            self.assertTrue(np.allclose(trajectory.velocities[-1], 0.0))
            self.assertEqual(trajectory.phases[-1], "stopped")
            self.assertEqual(trajectory.primitive_ids[-1], "stopped")
            self.assertEqual(trajectory.sequence_ids[0], "accelerate")
            self.assertTrue(
                all(
                    left != right
                    for left, right in zip(
                        trajectory.sequence_ids,
                        trajectory.sequence_ids[1:],
                    )
                )
            )
            follow_xy = np.linalg.norm(
                trajectory.positions[:, :2] - trajectory.goal_positions[:, :2],
                axis=1,
            )
            self.assertTrue(np.allclose(follow_xy, 1.0, atol=1e-6))
            self.assertLessEqual(
                float(np.linalg.norm(trajectory.goal_positions[:, :2], axis=1).max()),
                config.limits.max_horizontal_radius_m + 1e-6,
            )

    def test_turn_stage_always_contains_a_real_turn(self) -> None:
        config = self._load_pool("stage2_turn_pool_5hz_ratio_5to5.json")
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(7)
        for _ in range(16):
            trajectory = generator.sample(rng, [0.0, 0.0, -5.0], 0.0)
            self.assertIn("turn", trajectory.sequence_ids)
            turn_curvature = np.abs(
                trajectory.curvatures[
                    np.asarray(trajectory.primitive_ids, dtype=object) == "turn"
                ]
            )
            self.assertGreater(float(turn_curvature.max()), 0.01)


if __name__ == "__main__":
    unittest.main()
