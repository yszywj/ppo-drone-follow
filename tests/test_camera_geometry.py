from __future__ import annotations

import math
import unittest

import numpy as np

from pegasus_iris_fast_line_follow.camera_geometry import (
    CameraModelConfig,
    CameraQualityWindow,
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

    def test_behind_target_is_not_visible_and_features_are_finite(self) -> None:
        projection = self.project([-1.0, 0.2, 0.1])
        self.assertFalse(projection.in_front)
        self.assertFalse(projection.visible)
        self.assertFalse(projection.success_region)
        self.assertEqual(projection.center_quality, 0.0)
        self.assertTrue(np.all(np.isfinite(projection.actor_features(20.0))))

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


if __name__ == "__main__":
    unittest.main()
