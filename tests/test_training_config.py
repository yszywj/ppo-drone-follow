from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pegasus_iris_fast_line_follow.train_pegasus_iris_fast_line_follow_ppo import parse_args


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class TrainingConfigTest(unittest.TestCase):
    def _parse(self, config_name: str):
        config_path = PACKAGE_DIR / "configs" / config_name
        with patch.object(sys, "argv", ["train", "--config", str(config_path)]):
            return parse_args()

    def test_optional_task_description_is_only_emitted_when_configured(self) -> None:
        stage3 = self._parse("stage3_vertical_pool_5hz_ratio_5to5.json")
        stage4 = self._parse("stage4_combined_long_pool_5hz_ratio_5to5.json")
        self.assertFalse(hasattr(stage3, "task_description"))
        self.assertIn("combined long-horizon pool", stage4.task_description)

    def test_stage4_uses_flat_result_layout_and_fresh_seed(self) -> None:
        args = self._parse("stage4_combined_long_pool_5hz_ratio_5to5.json")
        self.assertEqual(args.seed, 1)
        self.assertTrue(args.reset_rng_on_load)
        self.assertEqual(Path(args.results_root).name, "ppo_train")

    def test_stage5_advances_seed_and_episode_length(self) -> None:
        args = self._parse("stage5_combined_extra_long_pool_400k_seed2.json")
        self.assertEqual(args.seed, 2)
        self.assertEqual(args.episode_length, 400)
        self.assertEqual(args.num_env_steps, 400000)

    def test_stage5a_uses_corrective_reward_intermediate_curriculum(self) -> None:
        args = self._parse(
            "stage5a_corrective_reward_intermediate_pool_100k_seed3.json"
        )
        self.assertEqual(args.seed, 3)
        self.assertEqual(args.rollout_steps, 128)
        self.assertEqual(args.episode_length, 350)
        self.assertEqual(args.num_env_steps, 100000)
        self.assertTrue(args.reset_optimizer)
        self.assertAlmostEqual(args.gamma, 0.997)
        self.assertAlmostEqual(args.reward_velocity_scale, 0.05)
        self.assertAlmostEqual(args.reward_position_recovery_scale, 0.5)
        self.assertAlmostEqual(args.reward_velocity_correction_gain, 0.5)
        self.assertAlmostEqual(args.reward_velocity_correction_max_mps, 0.25)

    def test_stage5b_rolls_back_and_removes_raw_action_magnitude_penalty(self) -> None:
        args = self._parse("stage5b_progress_reward_bridge_50k_seed5.json")
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.num_env_steps, 50000)
        self.assertEqual(args.policy_ratio, 0.5)
        self.assertTrue(args.reset_optimizer)
        self.assertEqual(args.lr_schedule, "constant")
        self.assertAlmostEqual(args.lr, 1e-6)
        self.assertAlmostEqual(args.critic_lr, 5e-6)
        self.assertAlmostEqual(args.reward_control_scale, 0.0)
        self.assertAlmostEqual(args.reward_action_delta_scale, 0.005)
        self.assertAlmostEqual(args.reward_moving_progress_scale, 3.0)
        self.assertAlmostEqual(args.reward_position_recovery_scale, 0.1)
        self.assertIn("actor_critic_update_10.pt", args.load_checkpoint)
        self.assertEqual(args.best_checkpoint_window, 5)

    def test_stage7_inherits_baseline_and_overrides_credit_assignment(self) -> None:
        args = self._parse(
            "stage7_local_credit_popart_5hz_ratio_5to5_100k_seed7.json"
        )
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.rollout_steps, 256)
        self.assertEqual(args.policy_ratio, 0.5)
        self.assertEqual(args.actor_recurrent_mode, "frozen")
        self.assertTrue(args.use_popart)
        self.assertAlmostEqual(args.gae_lambda, 0.98)
        self.assertAlmostEqual(args.reward_local_tracking_scale, 2.0)
        self.assertAlmostEqual(args.reference_kl_coef, 0.05)
        self.assertEqual(args.motion_pool_config["min_segments"], 6)
        self.assertEqual(args.motion_pool_config["max_segments"], 8)
        self.assertIn("cruise", args.motion_pool_config["primitive_library"])

    def test_fixed_evaluation_is_deterministic_and_does_not_train(self) -> None:
        args = self._parse("eval_fixed_seed5_best_50k.json")
        self.assertTrue(args.evaluation_only)
        self.assertTrue(args.deterministic_actions)
        self.assertTrue(args.independent_env_rng)
        self.assertEqual(args.actor_recurrent_mode, "frozen")
        self.assertEqual(args.best_checkpoint_window, 0)


if __name__ == "__main__":
    unittest.main()
