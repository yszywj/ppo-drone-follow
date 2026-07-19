from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from pegasus_iris_fast_line_follow.motion_task import MotionPoolConfig, MotionTaskGenerator
from pegasus_iris_fast_line_follow.training_config import load_training_config


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

    def test_vertical_stage_requires_bounded_climb_or_descent(self) -> None:
        config = self._load_pool("stage3_vertical_pool_5hz_ratio_5to5.json")
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(19)
        sampled_directions = set()
        for _ in range(64):
            trajectory = generator.sample(
                rng,
                [0.0, 0.0, -5.0],
                rng.uniform(-np.pi, np.pi),
            )
            vertical_sequence = [
                primitive_id
                for primitive_id in trajectory.sequence_ids
                if primitive_id in {"climb", "descend"}
            ]
            self.assertEqual(len(vertical_sequence), 1)
            vertical_ids = set(vertical_sequence)
            sampled_directions.update(vertical_ids)
            self.assertNotIn("turn", trajectory.sequence_ids)
            self.assertLessEqual(
                float(np.max(np.abs(trajectory.positions[:, 2] + 5.0))),
                config.limits.max_vertical_displacement_m + 1e-6,
            )
            primitive_ids = np.asarray(trajectory.primitive_ids, dtype=object)
            if "climb" in vertical_ids:
                climb_velocity = trajectory.velocities[primitive_ids == "climb", 2]
                self.assertLess(float(np.min(climb_velocity)), -0.03)
            if "descend" in vertical_ids:
                descend_velocity = trajectory.velocities[primitive_ids == "descend", 2]
                self.assertGreater(float(np.max(descend_velocity)), 0.03)
            self.assertTrue(np.allclose(trajectory.velocities[-1], 0.0))
            self.assertEqual(trajectory.phases[-1], "stopped")
        self.assertEqual(sampled_directions, {"climb", "descend"})

    def test_combined_long_stage_covers_all_curriculum_families(self) -> None:
        config_data = json.loads(
            (PACKAGE_DIR / "configs" / "stage4_combined_long_pool_5hz_ratio_5to5.json").read_text(
                encoding="utf-8"
            )
        )
        config = MotionPoolConfig.from_dict(config_data["task"]["motion_pool"])
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(1)
        sampled_lengths = set()
        sampled_primitives = set()
        for _ in range(64):
            trajectory = generator.sample(
                rng,
                [0.0, 0.0, -5.0],
                rng.uniform(-np.pi, np.pi),
            )
            sampled_lengths.add(len(trajectory.sequence_ids))
            sampled_primitives.update(trajectory.sequence_ids)
            self.assertEqual(trajectory.sequence_ids[0], "accelerate")
            self.assertIn("turn", trajectory.sequence_ids)
            self.assertTrue(
                {"climb", "descend"}.intersection(trajectory.sequence_ids)
            )
            self.assertLessEqual(
                trajectory.duration_sec,
                config_data["training"]["episode_length"]
                * config_data["environment"]["step_dt_sim_sec"],
            )
            self.assertLessEqual(
                float(np.linalg.norm(trajectory.positions[:, :2], axis=1).max()),
                config.limits.max_horizontal_radius_m + 1e-6,
            )
        self.assertEqual(sampled_lengths, {7, 8, 9})
        self.assertEqual(
            sampled_primitives,
            {"accelerate", "cruise", "decelerate", "turn", "climb", "descend"},
        )

    def test_extra_long_stage_samples_requested_segment_counts(self) -> None:
        config_data = json.loads(
            (PACKAGE_DIR / "configs" / "stage5_combined_extra_long_pool_400k_seed2.json").read_text(
                encoding="utf-8"
            )
        )
        config = MotionPoolConfig.from_dict(config_data["task"]["motion_pool"])
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(2)
        sampled_lengths = set()
        for _ in range(48):
            trajectory = generator.sample(
                rng,
                [0.0, 0.0, -5.0],
                rng.uniform(-np.pi, np.pi),
            )
            sampled_lengths.add(len(trajectory.sequence_ids))
            self.assertLessEqual(
                trajectory.duration_sec,
                config_data["training"]["episode_length"]
                * config_data["environment"]["step_dt_sim_sec"],
            )
        self.assertEqual(sampled_lengths, {11, 12, 13})

    def test_camera_stage_enforces_large_turn_and_vertical_amplitudes(self) -> None:
        _, special, _ = load_training_config(
            PACKAGE_DIR
            / "configs"
            / "stage8_camera_short_high_amplitude_5hz_ratio_5to5_100k_seed8.json"
        )
        config = MotionPoolConfig.from_dict(special["motion_pool"])
        generator = MotionTaskGenerator(config, dt=0.2, follow_distance_m=1.0)
        rng = np.random.default_rng(8)
        for _ in range(48):
            trajectory = generator.sample(
                rng,
                [0.0, 0.0, -5.0],
                rng.uniform(-math.pi / 3.0, math.pi / 3.0),
            )
            self.assertIn(len(trajectory.sequence_ids), {3, 4, 5})
            for segment_index, primitive_id in enumerate(
                trajectory.sequence_ids
            ):
                indices = np.flatnonzero(
                    trajectory.segment_indices == segment_index
                )
                start = max(0, int(indices[0]) - 1)
                end = int(indices[-1])
                if primitive_id == "turn":
                    heading = np.unwrap(trajectory.headings[start : end + 1])
                    change_deg = abs(np.degrees(heading[-1] - heading[0]))
                    self.assertGreaterEqual(change_deg + 1e-6, 35.0)
                if primitive_id in {"climb", "descend"}:
                    displacement = abs(
                        trajectory.positions[end, 2]
                        - trajectory.positions[start, 2]
                    )
                    self.assertGreaterEqual(displacement + 1e-6, 0.6)


if __name__ == "__main__":
    unittest.main()
