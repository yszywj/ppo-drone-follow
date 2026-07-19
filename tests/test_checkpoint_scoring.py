from __future__ import annotations

import json
import unittest

from pegasus_iris_fast_line_follow.checkpoint_scoring import (
    CheckpointScoreConfig,
    aggregate_checkpoint_score,
    violates_checkpoint_guardrails,
)


class CheckpointScoringTest(unittest.TestCase):
    def row(self, **overrides):
        row = {
            "completed_episode_count": 10,
            "success_count": 8,
            "timeout_count": 1,
            "other_done_count": 1,
            "mean_completed_final_xy_err": 0.2,
            "moving_good_sample_fraction": 0.8,
            "moving_xy_good_sample_fraction": 0.85,
            "stopped_xy_zone_fraction": 0.8,
            "stopped_position_zone_fraction": 0.75,
            "stopped_stationary_fraction": 0.9,
            "camera_good_sample_fraction": 0.9,
            "primitive_good_fraction": json.dumps({"final_stop": 0.8}),
        }
        row.update(overrides)
        return row

    def test_score_includes_completed_outcomes_and_final_stop(self) -> None:
        config = CheckpointScoreConfig(0.3, camera_enabled=True)
        good_score, good = aggregate_checkpoint_score([self.row()], config)
        bad_score, bad = aggregate_checkpoint_score(
            [
                self.row(
                    success_count=2,
                    timeout_count=6,
                    mean_completed_final_xy_err=0.8,
                    stopped_xy_zone_fraction=0.2,
                    camera_good_sample_fraction=0.3,
                    primitive_good_fraction=json.dumps({"final_stop": 0.2}),
                )
            ],
            config,
        )
        self.assertGreater(good_score, bad_score)
        self.assertAlmostEqual(good["overall_success"], 0.8)
        self.assertAlmostEqual(good["final_stop"], 0.8)
        self.assertGreater(good["final_xy_quality"], bad["final_xy_quality"])

    def test_outcome_rates_are_weighted_by_completed_episodes(self) -> None:
        config = CheckpointScoreConfig(0.3)
        _, components = aggregate_checkpoint_score(
            [
                self.row(completed_episode_count=2, success_count=2, timeout_count=0),
                self.row(completed_episode_count=8, success_count=0, timeout_count=8),
            ],
            config,
        )
        self.assertAlmostEqual(components["overall_success"], 0.2)
        self.assertAlmostEqual(components["timeout_rate"], 0.8)

    def test_guardrail_rejects_stopped_regression(self) -> None:
        config = CheckpointScoreConfig(0.3, camera_enabled=True, guardrail_drop=0.08)
        incumbent = {
            "overall_success": 0.8,
            "stopped_xy": 0.8,
            "final_stop": 0.8,
            "final_xy_quality": 0.8,
            "camera_visibility": 0.9,
        }
        candidate = dict(incumbent, stopped_xy=0.7)
        self.assertTrue(
            violates_checkpoint_guardrails(candidate, incumbent, config)
        )


if __name__ == "__main__":
    unittest.main()
