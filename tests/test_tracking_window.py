from __future__ import annotations

import unittest

from pegasus_iris_fast_line_follow.tracking_window import (
    TrackingQualityWindow,
    seconds_to_steps,
    tracking_event_rewards,
)


class TrackingQualityWindowTest(unittest.TestCase):
    def test_seconds_to_steps_rounds_up_and_validates_step_dt(self) -> None:
        self.assertEqual(seconds_to_steps(2.0, 0.2), 10)
        self.assertEqual(seconds_to_steps(2.01, 0.2), 11)
        self.assertEqual(seconds_to_steps(0.0, 0.2), 1)
        self.assertEqual(seconds_to_steps(-1.0, 0.2), 1)
        with self.assertRaises(ValueError):
            seconds_to_steps(2.0, 0.0)

    def make_window(self) -> TrackingQualityWindow:
        return TrackingQualityWindow(
            window_steps=4,
            interval_steps=4,
            xy_tolerance_m=0.45,
            velocity_tolerance_mps=0.25,
            z_tolerance_m=0.40,
            xy_sigma_m=0.50,
            velocity_sigma_mps=0.25,
            z_sigma_m=0.40,
        )

    def test_reward_event_requires_a_complete_interval(self) -> None:
        window = self.make_window()
        for _ in range(3):
            snapshot = window.append(0.1, 0.05, 0.1)
            self.assertFalse(snapshot.ready)
        snapshot = window.append(0.1, 0.05, 0.1)
        self.assertTrue(snapshot.ready)
        self.assertTrue(snapshot.reward_event)
        self.assertEqual(snapshot.xy_good_fraction, 1.0)

    def test_overlapping_history_does_not_emit_every_step(self) -> None:
        window = self.make_window()
        events = [window.append(0.1, 0.05, 0.1).reward_event for _ in range(8)]
        self.assertEqual(events, [False, False, False, True, False, False, False, True])

    def test_drift_and_quality_distinguish_sustained_departure(self) -> None:
        good = self.make_window()
        bad = self.make_window()
        for error in (0.20, 0.18, 0.16, 0.14):
            good_snapshot = good.append(error, 0.05, 0.10)
        for error in (0.20, 0.35, 0.50, 0.65):
            bad_snapshot = bad.append(error, 0.30, 0.10)
        self.assertLess(good_snapshot.drift_delta_m, 0.0)
        self.assertGreater(bad_snapshot.drift_delta_m, 0.0)
        self.assertGreater(
            good_snapshot.soft_joint_quality,
            bad_snapshot.soft_joint_quality,
        )

    def test_reset_removes_previous_trajectory_history(self) -> None:
        window = self.make_window()
        for _ in range(4):
            window.append(0.1, 0.05, 0.1)
        window.reset()
        self.assertFalse(window.append(0.1, 0.05, 0.1).ready)

    def test_event_reward_blends_quality_and_penalizes_only_departure(self) -> None:
        good_window = self.make_window()
        bad_window = self.make_window()
        for error in (0.2, 0.18, 0.16, 0.14):
            good = good_window.append(error, 0.05, 0.1)
        for error in (0.2, 0.35, 0.5, 0.65):
            bad = bad_window.append(error, 0.3, 0.1)
        good_tracking, good_drift = tracking_event_rewards(
            good, 0.3, 0.05, 0.45, 2.0, 2.0
        )
        bad_tracking, bad_drift = tracking_event_rewards(
            bad, 0.3, 0.05, 0.45, 2.0, 2.0
        )
        self.assertGreater(good_tracking, bad_tracking)
        self.assertEqual(good_drift, 0.0)
        self.assertLess(bad_drift, 0.0)


if __name__ == "__main__":
    unittest.main()
