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
            obs_dim=32,
            action_dim=4,
            critic_obs_dim=77,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64, 32),
            recurrent_hidden_size=32,
            temporal_gate_init=0.0,
        )

    def test_zero_gates_preserve_frame_policy_output(self) -> None:
        obs = torch.randn(3, 32)
        hidden_a = self.policy.initial_state(3)
        hidden_b = torch.randn_like(hidden_a)
        action_a, _ = self.policy.deterministic_action(obs, hidden_a)
        action_b, _ = self.policy.deterministic_action(obs, hidden_b)
        self.assertTrue(torch.allclose(action_a, action_b, atol=1e-7))

    def test_recurrent_sequence_shapes_and_gradients(self) -> None:
        time_steps = 5
        batch_size = 4
        obs = torch.randn(time_steps, batch_size, 32)
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
        self.assertIsNotNone(self.policy.temporal_gate.grad)
        self.assertIsNotNone(self.policy.critic.weight.grad)

    def test_nonzero_gate_trains_recurrent_branch(self) -> None:
        with torch.no_grad():
            self.policy.temporal_gate.fill_(0.05)
        obs = torch.randn(3, 32)
        action, _ = self.policy.deterministic_action(
            obs,
            torch.randn(3, self.policy.recurrent_hidden_size),
        )
        action.square().mean().backward()
        recurrent_grad = self.policy.actor_gru.weight_ih.grad
        self.assertIsNotNone(recurrent_grad)
        self.assertGreater(float(recurrent_grad.abs().sum()), 0.0)

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
                obs_dim=32,
                action_dim=4,
                critic_obs_dim=77,
                actor_hidden_sizes=(64, 32),
                critic_hidden_sizes=(64, 32),
                recurrent_hidden_size=32,
                temporal_gate_init=0.0,
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

    def test_partial_load_skips_removed_actor_reference_branch(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_checkpoint.pt"
            legacy_state = dict(self.policy.state_dict())
            legacy_state["reference_encoder.0.weight"] = torch.randn(8, 15)
            legacy_state["reference_gate"] = torch.tensor(0.2)
            torch.save({"state_dict": legacy_state}, checkpoint)
            restored = ActorCritic(
                obs_dim=32,
                action_dim=4,
                critic_obs_dim=77,
                actor_hidden_sizes=(64, 32),
                critic_hidden_sizes=(64, 32),
                recurrent_hidden_size=32,
            )
            load_transplanted_checkpoint(
                restored,
                checkpoint,
                torch.device("cpu"),
                allow_partial=True,
            )
            self.assertTrue(
                torch.allclose(self.policy.actor_mean.weight, restored.actor_mean.weight)
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_load_keeps_rng_payload_on_cpu(self) -> None:
        device = torch.device("cuda:0")
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cuda_checkpoint.pt"
            actor_optimizer = torch.optim.Adam(
                list(self.policy.actor_parameters()),
                lr=1e-4,
            )
            critic_optimizer = torch.optim.Adam(
                list(self.policy.critic_parameters()),
                lr=2e-4,
            )
            save_policy_checkpoint(
                checkpoint,
                self.policy,
                update=1,
                total_steps=64,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            )
            restored = ActorCritic(
                obs_dim=32,
                action_dim=4,
                critic_obs_dim=77,
                actor_hidden_sizes=(64, 32),
                critic_hidden_sizes=(64, 32),
                recurrent_hidden_size=32,
            ).to(device)
            payload = load_transplanted_checkpoint(
                restored,
                checkpoint,
                device,
            )
            self.assertTrue(
                all(
                    state.device.type == "cpu"
                    for state in payload["torch_cuda_random_state"]
                )
            )
            torch.cuda.set_rng_state_all(payload["torch_cuda_random_state"])


if __name__ == "__main__":
    unittest.main()
