from __future__ import annotations

import unittest

import numpy as np

from pegasus_iris_fast_line_follow.reward_shaping import (
    clipped_error_progress,
    corrective_velocity_reference,
    position_reward_qualities,
)


class RewardShapingTest(unittest.TestCase):
    def test_velocity_reference_matches_goal_at_correct_position(self) -> None:
        reference, correction_speed = corrective_velocity_reference(
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            gain=0.5,
            max_correction_speed=0.25,
        )
        np.testing.assert_allclose(reference, [0.4, 0.0, 0.0])
        self.assertEqual(correction_speed, 0.0)

    def test_velocity_reference_closes_error_and_respects_limit(self) -> None:
        reference, correction_speed = corrective_velocity_reference(
            [0.4, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            gain=0.5,
            max_correction_speed=0.25,
        )
        np.testing.assert_allclose(reference, [0.65, 0.0, 0.0])
        self.assertAlmostEqual(correction_speed, 0.25)

    def test_recovery_quality_remains_informative_at_one_meter(self) -> None:
        precise, recovery = position_reward_qualities(
            xy_error=1.0,
            z_error=0.0,
            position_sigma=0.5,
            z_sigma=0.4,
            recovery_position_sigma=1.5,
            recovery_z_sigma=0.8,
        )
        self.assertLess(precise, 0.02)
        self.assertGreater(recovery, 0.65)

    def test_error_progress_rewards_recovery_and_penalizes_drift(self) -> None:
        self.assertAlmostEqual(clipped_error_progress(0.6, 0.5, 0.2), 0.1)
        self.assertAlmostEqual(clipped_error_progress(0.5, 0.6, 0.2), -0.1)

    def test_error_progress_is_clipped_symmetrically(self) -> None:
        self.assertAlmostEqual(clipped_error_progress(1.0, 0.2, 0.15), 0.15)
        self.assertAlmostEqual(clipped_error_progress(0.2, 1.0, 0.15), -0.15)


if __name__ == "__main__":
    unittest.main()
