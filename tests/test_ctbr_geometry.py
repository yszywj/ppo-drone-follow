from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus_iris_fast_line_follow.control_geometry import (
    mix_ctbr_commands,
    quaternion_ned_frd_to_euler,
    vehicle_yaw_for_target_bearing,
    wrap_angle_pi,
    xy_tracking_attitude_targets,
)


class CTBRMixTest(unittest.TestCase):
    def test_ratio_mode_is_an_exact_fixed_convex_mix(self) -> None:
        helper = np.array([0.08, -0.04, 0.3, 0.66])
        policy = np.array([-0.02, 0.06, -0.1, 0.54])
        actual = mix_ctbr_commands(helper, policy, policy_ratio=0.3)
        np.testing.assert_allclose(
            actual,
            0.7 * helper + 0.3 * policy,
            atol=1e-15,
            rtol=0.0,
        )

    def test_neutral_policy_does_not_restore_full_helper_authority(self) -> None:
        helper = np.array([0.08, -0.04, 0.6, 0.66])
        neutral_policy = np.array([0.0, 0.0, 0.0, 0.6])
        actual = mix_ctbr_commands(helper, neutral_policy, policy_ratio=0.3)
        np.testing.assert_allclose(
            actual,
            [0.056, -0.028, 0.42, 0.642],
            atol=1e-15,
            rtol=0.0,
        )
        self.assertFalse(np.array_equal(actual, helper))

    def test_invalid_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mix_ctbr_commands(np.zeros(4), np.zeros(4), policy_ratio=1.01)


class AerospaceAttitudeTest(unittest.TestCase):
    def test_zyx_decomposition_does_not_mix_yaw_into_roll_pitch(self) -> None:
        expected_roll = math.radians(5.0)
        expected_pitch = math.radians(-7.0)
        for yaw_deg in (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0):
            expected_yaw = math.radians(yaw_deg)
            quaternion = Rotation.from_euler(
                "ZYX",
                [expected_yaw, expected_pitch, expected_roll],
            ).as_quat()
            roll, pitch, yaw = quaternion_ned_frd_to_euler(quaternion)
            self.assertAlmostEqual(roll, expected_roll, places=7)
            self.assertAlmostEqual(pitch, expected_pitch, places=7)
            self.assertAlmostEqual(yaw, expected_yaw, places=7)

    def test_invalid_quaternion_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quaternion_ned_frd_to_euler([0.0, 0.0, 1.0])

    def test_vehicle_yaw_places_world_target_at_requested_bearing(self) -> None:
        for line_yaw_deg in (-180.0, -120.0, -15.0, 0.0, 75.0, 179.0):
            for bearing_deg in (-40.0, -10.0, 0.0, 25.0, 40.0):
                line_yaw = math.radians(line_yaw_deg)
                bearing = math.radians(bearing_deg)
                vehicle_yaw = vehicle_yaw_for_target_bearing(
                    line_yaw,
                    bearing,
                )
                actual = wrap_angle_pi(line_yaw - vehicle_yaw)
                self.assertAlmostEqual(actual, bearing, places=12)


class HelperXYGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gains = {
            "goal_feedback_scale": 1.0,
            "position_gain": 0.04,
            "velocity_damping_gain": 0.18,
            "target_velocity_gain": 0.18,
            "target_acceleration_gain": 0.102,
            "max_tilt_cmd": 0.12,
            "control_mode": "body",
        }

    @staticmethod
    def rotate(vector: np.ndarray, yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array(
            [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]],
            dtype=np.float64,
        )

    def test_body_targets_are_invariant_to_world_heading(self) -> None:
        vectors = (
            np.array([-0.8, 0.25]),
            np.array([0.1, -0.05]),
            np.array([0.45, 0.08]),
            np.array([0.2, -0.1]),
        )
        baseline = xy_tracking_attitude_targets(
            *vectors,
            current_yaw=0.0,
            **self.gains,
        )
        for yaw_deg in (-120.0, -60.0, -20.0, 35.0, 90.0, 150.0):
            yaw = math.radians(yaw_deg)
            rotated = tuple(self.rotate(vector, yaw) for vector in vectors)
            actual = xy_tracking_attitude_targets(
                *rotated,
                current_yaw=yaw,
                **self.gains,
            )
            np.testing.assert_allclose(actual, baseline, atol=1e-12, rtol=0.0)

    def test_goal_ahead_commands_forward_tilt(self) -> None:
        roll, pitch = xy_tracking_attitude_targets(
            [-1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            current_yaw=0.0,
            **self.gains,
        )
        self.assertAlmostEqual(roll, 0.0)
        self.assertLess(pitch, 0.0)


if __name__ == "__main__":
    unittest.main()
