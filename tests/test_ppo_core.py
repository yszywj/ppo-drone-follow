from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pegasus_iris_fast_line_follow.ppo_core import (
    ActorCritic,
    load_transplanted_checkpoint,
    restore_training_state,
    save_policy_checkpoint,
)


class RecurrentActorCriticTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.policy = ActorCritic(
            obs_dim=47,
            action_dim=4,
            critic_obs_dim=77,
            base_obs_dim=32,
            reference_dim=15,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64, 32),
            recurrent_hidden_size=32,
            reference_hidden_size=32,
        )

    def test_zero_gates_preserve_frame_policy_output(self) -> None:
        base = torch.randn(3, 32)
        obs_a = torch.cat((base, torch.zeros(3, 15)), dim=1)
        obs_b = torch.cat((base, torch.randn(3, 15)), dim=1)
        hidden_a = self.policy.initial_state(3)
        hidden_b = torch.randn_like(hidden_a)
        action_a, _ = self.policy.deterministic_action(obs_a, hidden_a)
        action_b, _ = self.policy.deterministic_action(obs_b, hidden_b)
        self.assertTrue(torch.allclose(action_a, action_b, atol=1e-7))

    def test_recurrent_sequence_shapes_and_gradients(self) -> None:
        time_steps = 5
        batch_size = 4
        obs = torch.randn(time_steps, batch_size, 47)
        critic_obs = torch.randn(time_steps, batch_size, 77)
        actions = torch.tanh(torch.randn(time_steps, batch_size, 4))
        episode_starts = torch.zeros(time_steps, batch_size)
        episode_starts[0] = 1.0
        log_prob, entropy, value, hidden = self.policy.evaluate_actions_sequence(
            obs,
            critic_obs,
            actions,
            self.policy.initial_state(batch_size),
            episode_starts,
        )
        self.assertEqual(tuple(log_prob.shape), (time_steps, batch_size))
        self.assertEqual(tuple(entropy.shape), (time_steps, batch_size))
        self.assertEqual(tuple(value.shape), (time_steps, batch_size))
        self.assertEqual(tuple(hidden.shape), (batch_size, 32))
        loss = -(log_prob.mean() + 0.01 * entropy.mean()) + value.square().mean()
        loss.backward()
        self.assertIsNotNone(self.policy.actor_mean.weight.grad)
        self.assertIsNotNone(self.policy.critic.weight.grad)

    def test_checkpoint_round_trip_restores_optimizers(self) -> None:
        actor_optimizer = torch.optim.Adam(list(self.policy.actor_parameters()), lr=1e-4)
        critic_optimizer = torch.optim.Adam(list(self.policy.critic_parameters()), lr=2e-4)
        actor_optimizer.zero_grad()
        self.policy.log_std.square().sum().backward()
        actor_optimizer.step()
        rng = np.random.default_rng(123)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_policy_checkpoint(
                checkpoint,
                self.policy,
                update=3,
                total_steps=1024,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                env_rng=rng,
            )
            restored = ActorCritic(
                obs_dim=47,
                action_dim=4,
                critic_obs_dim=77,
                base_obs_dim=32,
                reference_dim=15,
                actor_hidden_sizes=(64, 32),
                critic_hidden_sizes=(64, 32),
                recurrent_hidden_size=32,
                reference_hidden_size=32,
            )
            restored_actor_optimizer = torch.optim.Adam(
                list(restored.actor_parameters()),
                lr=1e-4,
            )
            restored_critic_optimizer = torch.optim.Adam(
                list(restored.critic_parameters()),
                lr=2e-4,
            )
            payload = load_transplanted_checkpoint(
                restored,
                checkpoint,
                torch.device("cpu"),
            )
            self.assertTrue(
                restore_training_state(
                    payload,
                    restored_actor_optimizer,
                    restored_critic_optimizer,
                    env_rng=np.random.default_rng(),
                )
            )
            self.assertEqual(payload["format_version"], 3)
            self.assertTrue(
                torch.allclose(self.policy.log_std, restored.log_std)
            )


if __name__ == "__main__":
    unittest.main()
