from __future__ import annotations

import math
import unittest

import numpy as np

from pegasus_iris_fast_line_follow.camera_geometry import (
    CAMERA_OBSERVATION_DIM,
    CameraModelConfig,
    CameraQualityWindow,
    camera_centering_yaw_rate,
    camera_observation_vector,
    project_target_to_camera,
)


class CameraGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CameraModelConfig(
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
            success_margin=0.9,
        )
        self.identity = np.asarray([0.0, 0.0, 0.0, 1.0])

    def project(self, target, config=None):
        return project_target_to_camera(
            [0.0, 0.0, 0.0],
            self.identity,
            target,
            config or self.config,
        )

    def test_forward_target_projects_to_image_center(self) -> None:
        projection = self.project([2.0, 0.0, 0.0])
        self.assertTrue(projection.visible)
        self.assertTrue(projection.success_region)
        self.assertAlmostEqual(projection.normalized_u, 0.0)
        self.assertAlmostEqual(projection.normalized_v, 0.0)
        self.assertAlmostEqual(projection.center_quality, 1.0)

    def test_horizontal_and_vertical_fov_edges(self) -> None:
        horizontal = self.project([1.0, 1.0, 0.0])
        self.assertTrue(horizontal.visible)
        self.assertFalse(horizontal.success_region)
        self.assertAlmostEqual(horizontal.normalized_u, 1.0, places=6)
        vertical_edge = math.tan(math.radians(30.0))
        vertical = self.project([1.0, 0.0, vertical_edge])
        self.assertTrue(vertical.visible)
        self.assertAlmostEqual(vertical.normalized_v, 1.0, places=6)

    def test_body_yaw_is_applied_before_projection(self) -> None:
        body_to_ned = np.asarray(
            [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
        )
        projection = project_target_to_camera(
            [0.0, 0.0, 0.0],
            body_to_ned,
            [0.0, 2.0, 0.0],
            self.config,
        )
        self.assertTrue(projection.success_region)
        self.assertAlmostEqual(projection.normalized_u, 0.0, places=6)

    def test_positive_mount_pitch_points_camera_down(self) -> None:
        config = CameraModelConfig(
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
            mount_pitch_down_deg=30.0,
        )
        target = [math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0))]
        projection = self.project(target, config)
        self.assertTrue(projection.success_region)
        self.assertAlmostEqual(projection.normalized_v, 0.0, places=6)

    def test_behind_target_is_not_visible(self) -> None:
        projection = self.project([-1.0, 0.2, 0.1])
        self.assertFalse(projection.in_front)
        self.assertFalse(projection.visible)
        self.assertFalse(projection.success_region)
        self.assertEqual(projection.center_quality, 0.0)

    def test_near_and_far_planes_use_optical_axis_depth(self) -> None:
        config = CameraModelConfig(
            horizontal_fov_deg=170.0,
            vertical_fov_deg=170.0,
            near_clip_m=0.1,
            far_clip_m=2.0,
        )
        inside = self.project([1.9, 1.0, 0.0], config)
        beyond = self.project([2.1, 0.0, 0.0], config)
        self.assertTrue(inside.visible)
        self.assertGreater(inside.range_m, 2.0)
        self.assertFalse(beyond.visible)

    def test_detector_observation_is_normalized_and_visibility_masked(self) -> None:
        centered = self.project([2.0, 0.0, 0.0])
        features = camera_observation_vector(centered, self.config)
        self.assertEqual(features.shape, (CAMERA_OBSERVATION_DIM,))
        self.assertAlmostEqual(float(features[0]), 0.0)
        self.assertAlmostEqual(float(features[1]), 0.0)
        self.assertGreater(float(features[2]), 0.0)
        self.assertGreater(float(features[3]), 0.0)
        self.assertEqual(float(features[4]), 1.0)
        self.assertEqual(float(features[5]), 1.0)

        hidden = camera_observation_vector(
            self.project([-1.0, 0.2, 0.1]),
            self.config,
        )
        self.assertTrue(np.array_equal(hidden, np.zeros(6, dtype=np.float32)))


class CameraQualityWindowTest(unittest.TestCase):
    def test_fixed_window_fraction_and_interval(self) -> None:
        window = CameraQualityWindow(window_steps=4, interval_steps=2)
        snapshots = [
            window.append(True, good, quality)
            for good, quality in (
                (True, 1.0),
                (False, 0.2),
                (True, 0.8),
                (True, 0.6),
            )
        ]
        self.assertFalse(snapshots[2].ready)
        final = snapshots[-1]
        self.assertTrue(final.ready)
        self.assertTrue(final.reward_event)
        self.assertAlmostEqual(final.success_fraction, 0.75)
        self.assertAlmostEqual(final.mean_center_quality, 0.65)
        self.assertAlmostEqual(final.joint_quality, 0.4875)


class CameraYawHelperTest(unittest.TestCase):
    def command(self, bearing: float, yaw_rate: float = 0.0) -> float:
        return camera_centering_yaw_rate(
            bearing,
            yaw_rate,
            proportional_gain=1.0,
            damping_gain=0.15,
            max_rate_rad_s=0.6,
            deadband_rad=math.radians(2.0),
        )

    def test_turns_toward_target_on_both_sides(self) -> None:
        self.assertGreater(self.command(math.radians(20.0)), 0.0)
        self.assertLess(self.command(math.radians(-20.0)), 0.0)

    def test_saturates_for_target_behind_camera(self) -> None:
        self.assertAlmostEqual(self.command(math.radians(170.0)), 0.6)
        self.assertAlmostEqual(self.command(math.radians(-170.0)), -0.6)

    def test_center_deadband_keeps_rate_damping(self) -> None:
        self.assertAlmostEqual(self.command(math.radians(1.0)), 0.0)
        self.assertLess(self.command(0.0, yaw_rate=0.4), 0.0)


if __name__ == "__main__":
    unittest.main()
