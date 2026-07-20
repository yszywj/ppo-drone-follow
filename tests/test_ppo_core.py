from __future__ import annotations

import unittest
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pegasus_iris_fast_line_follow.ppo_core import (
    ActorCritic,
    PopArtValueNormalizer,
    capture_policy_checkpoint,
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
            critic_obs_dim=83,
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
        critic_obs = torch.randn(time_steps, batch_size, 83)
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

    def test_actor_output_reset_preserves_features_and_critic(self) -> None:
        preserved = {
            key: value.detach().clone()
            for key, value in self.policy.state_dict().items()
            if key not in {"actor_mean.weight", "actor_mean.bias", "log_std"}
        }
        with torch.no_grad():
            self.policy.actor_mean.weight.fill_(1.0)
            self.policy.actor_mean.bias.fill_(1.0)
            self.policy.log_std.zero_()

        self.policy.reset_actor_output(0.12)

        self.assertTrue(
            torch.allclose(
                self.policy.actor_mean.bias,
                torch.zeros_like(self.policy.actor_mean.bias),
            )
        )
        self.assertTrue(
            torch.allclose(
                self.policy.log_std,
                torch.full_like(self.policy.log_std, float(np.log(0.12))),
            )
        )
        self.assertFalse(
            torch.allclose(
                self.policy.actor_mean.weight,
                torch.ones_like(self.policy.actor_mean.weight),
            )
        )
        current = self.policy.state_dict()
        for key, expected in preserved.items():
            self.assertTrue(torch.equal(current[key], expected), key)

    def test_checkpoint_round_trip_restores_optimizers(self) -> None:
        actor_optimizer = torch.optim.Adam(list(self.policy.actor_parameters()), lr=1e-4)
        critic_optimizer = torch.optim.Adam(list(self.policy.critic_parameters()), lr=2e-4)
        reference_policy = copy.deepcopy(self.policy)
        with torch.no_grad():
            reference_policy.actor_mean.bias.add_(0.25)
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
                reference_policy=reference_policy,
            )
            restored = ActorCritic(
                obs_dim=32,
                action_dim=4,
                critic_obs_dim=83,
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
            self.assertEqual(payload["format_version"], 5)
            self.assertTrue(
                torch.allclose(
                    payload["reference_policy_state_dict"]["actor_mean.bias"],
                    reference_policy.actor_mean.bias,
                )
            )
            self.assertTrue(
                torch.allclose(self.policy.log_std, restored.log_std)
            )
            kept_rng = np.random.default_rng(999)
            expected_rng = np.random.default_rng(999)
            self.assertTrue(
                restore_training_state(
                    payload,
                    restored_actor_optimizer,
                    restored_critic_optimizer,
                    env_rng=kept_rng,
                    restore_rng=False,
                )
            )
            self.assertEqual(
                int(kept_rng.integers(0, 1_000_000)),
                int(expected_rng.integers(0, 1_000_000)),
            )

    def test_captured_checkpoint_is_not_changed_by_next_update(self) -> None:
        actor_optimizer = torch.optim.Adam(
            list(self.policy.actor_parameters()),
            lr=1e-4,
        )
        critic_optimizer = torch.optim.Adam(
            list(self.policy.critic_parameters()),
            lr=2e-4,
        )
        payload = capture_policy_checkpoint(
            self.policy,
            update=4,
            total_steps=2048,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )
        captured_bias = payload["state_dict"]["actor_mean.bias"].clone()
        with torch.no_grad():
            self.policy.actor_mean.bias.add_(0.5)
        self.assertTrue(
            torch.allclose(
                payload["state_dict"]["actor_mean.bias"],
                captured_bias,
            )
        )
        self.assertFalse(
            torch.allclose(self.policy.actor_mean.bias, captured_bias)
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
                critic_obs_dim=83,
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

    def test_partial_load_zeros_appended_observation_columns(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old_observation_checkpoint.pt"
            torch.save({"state_dict": self.policy.state_dict()}, checkpoint)
            restored = ActorCritic(
                obs_dim=38,
                action_dim=4,
                critic_obs_dim=89,
                actor_hidden_sizes=(64, 32),
                critic_hidden_sizes=(64, 32),
                recurrent_hidden_size=32,
                temporal_gate_init=0.0,
            )
            load_transplanted_checkpoint(
                restored,
                checkpoint,
                torch.device("cpu"),
                allow_partial=True,
            )
            self.assertTrue(
                torch.allclose(restored.actor_body[0].weight[:, 32:], torch.zeros(64, 6))
            )
            self.assertTrue(
                torch.allclose(restored.critic_body[0].weight[:, 83:], torch.zeros(64, 6))
            )
            old_obs = torch.randn(3, 32)
            new_obs = torch.cat((old_obs, torch.randn(3, 6)), dim=1)
            hidden = self.policy.initial_state(3)
            old_action, _ = self.policy.deterministic_action(old_obs, hidden)
            new_action, _ = restored.deterministic_action(new_obs, hidden)
            self.assertTrue(torch.allclose(old_action, new_action, atol=1e-7))
            old_critic_obs = torch.randn(3, 83)
            new_critic_obs = torch.cat(
                (old_critic_obs, torch.randn(3, 6)), dim=1
            )
            self.assertTrue(
                torch.allclose(
                    self.policy.value(old_critic_obs),
                    restored.value(new_critic_obs),
                    atol=1e-7,
                )
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
                critic_obs_dim=83,
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

    def test_distribution_sequence_reference_kl(self) -> None:
        obs = torch.randn(6, 3, 32)
        episode_starts = torch.zeros(6, 3)
        episode_starts[0] = 1.0
        reference = copy.deepcopy(self.policy)
        mean, std, _ = self.policy.distribution_parameters_sequence(
            obs,
            self.policy.initial_state(3),
            episode_starts,
        )
        reference_mean, reference_std, _ = (
            reference.distribution_parameters_sequence(
                obs,
                reference.initial_state(3),
                episode_starts,
            )
        )
        same_kl = (
            torch.log(reference_std / std)
            + (std.square() + (mean - reference_mean).square())
            / (2.0 * reference_std.square())
            - 0.5
        ).sum(dim=-1).mean()
        self.assertAlmostEqual(float(same_kl), 0.0, places=7)
        with torch.no_grad():
            self.policy.actor_mean.bias.add_(0.1)
        shifted_mean, shifted_std, _ = (
            self.policy.distribution_parameters_sequence(
                obs,
                self.policy.initial_state(3),
                episode_starts,
            )
        )
        shifted_kl = (
            torch.log(reference_std / shifted_std)
            + (
                shifted_std.square()
                + (shifted_mean - reference_mean).square()
            )
            / (2.0 * reference_std.square())
            - 0.5
        ).sum(dim=-1).mean()
        self.assertGreater(float(shifted_kl), 0.0)


class PopArtValueNormalizerTest(unittest.TestCase):
    def test_update_preserves_denormalized_value_predictions(self) -> None:
        torch.manual_seed(11)
        policy = ActorCritic(
            obs_dim=8,
            action_dim=2,
            critic_obs_dim=12,
            actor_hidden_sizes=(16, 8),
            critic_hidden_sizes=(16, 8),
            recurrent_hidden_size=8,
        )
        normalizer = PopArtValueNormalizer(torch.device("cpu"), beta=0.9)
        optimizer = torch.optim.Adam(policy.critic_parameters(), lr=1e-3)
        critic_obs = torch.randn(32, 12)
        raw_before = normalizer.denormalize(policy.value(critic_obs)).detach()
        returns = 20.0 + 7.0 * torch.randn(128)
        normalizer.update(returns, policy.critic, optimizer=optimizer)
        raw_after = normalizer.denormalize(policy.value(critic_obs)).detach()
        self.assertTrue(torch.allclose(raw_before, raw_after, atol=2e-5))
        normalized_returns = normalizer.normalize(returns)
        self.assertAlmostEqual(float(normalized_returns.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(normalized_returns.std(unbiased=False)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
