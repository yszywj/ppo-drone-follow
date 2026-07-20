from __future__ import annotations

import csv
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def mlp(input_dim: int, hidden_sizes: Sequence[int], activation: str) -> nn.Sequential:
    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one layer")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported activation: {activation}")
    layers: List[nn.Module] = []
    previous_dim = int(input_dim)
    activation_type = ACTIVATIONS[activation]
    for hidden_dim in hidden_sizes:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden layer sizes must be positive")
        linear = nn.Linear(previous_dim, hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.constant_(linear.bias, 0.0)
        layers.extend((linear, activation_type()))
        previous_dim = hidden_dim
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0
    ACTION_EPS = 1e-6

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_obs_dim: int | None = None,
        actor_hidden_sizes: Sequence[int] = (256, 256, 128),
        critic_hidden_sizes: Sequence[int] = (256, 256, 128),
        recurrent_hidden_size: int = 128,
        temporal_gate_init: float = 0.0,
        activation: str = "elu",
        init_std: float = 0.20,
    ):
        super().__init__()
        if init_std <= 0.0:
            raise ValueError("init_std must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.critic_obs_dim = int(critic_obs_dim if critic_obs_dim is not None else obs_dim)
        self.actor_hidden_sizes = tuple(int(size) for size in actor_hidden_sizes)
        self.critic_hidden_sizes = tuple(int(size) for size in critic_hidden_sizes)
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        if self.recurrent_hidden_size <= 0:
            raise ValueError("recurrent hidden size must be positive")
        self.activation = str(activation)
        self.actor_body = mlp(self.obs_dim, self.actor_hidden_sizes, self.activation)
        actor_latent_dim = self.actor_hidden_sizes[-1]
        self.actor_gru = nn.GRUCell(actor_latent_dim, self.recurrent_hidden_size)
        self.temporal_projection = nn.Linear(self.recurrent_hidden_size, actor_latent_dim)
        self.temporal_gate = nn.Parameter(
            torch.tensor(float(temporal_gate_init), dtype=torch.float32)
        )
        self.actor_mean = nn.Linear(actor_latent_dim, self.action_dim)
        self.critic_body = mlp(self.critic_obs_dim, self.critic_hidden_sizes, self.activation)
        self.critic = nn.Linear(self.critic_hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_std)), dtype=torch.float32))
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.temporal_projection.weight, gain=0.01)
        nn.init.constant_(self.temporal_projection.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def reset_actor_output(self, init_std: float) -> None:
        """Reset the CTBR residual head while preserving actor features and Critic."""
        if float(init_std) <= 0.0:
            raise ValueError("actor action standard deviation must be positive")
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        with torch.no_grad():
            self.log_std.fill_(float(np.log(init_std)))

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.recurrent_hidden_size, dtype=torch.float32, device=device)

    def actor_backbone_parameters(self):
        for module in (self.actor_body, self.actor_mean):
            yield from module.parameters()
        yield self.log_std

    def actor_recurrent_parameters(self):
        for module in (self.actor_gru, self.temporal_projection):
            yield from module.parameters()
        yield self.temporal_gate

    def actor_parameters(self):
        yield from self.actor_backbone_parameters()
        yield from self.actor_recurrent_parameters()

    def critic_parameters(self):
        yield from self.critic_body.parameters()
        yield from self.critic.parameters()

    def _actor_features_step(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"actor observation must have shape (batch, {self.obs_dim}), got {tuple(obs.shape)}")
        if hidden_state.shape != (obs.shape[0], self.recurrent_hidden_size):
            raise ValueError(
                "hidden state shape mismatch: expected "
                f"({obs.shape[0]}, {self.recurrent_hidden_size}), got {tuple(hidden_state.shape)}"
            )
        if episode_start is not None:
            hidden_state = hidden_state * (1.0 - episode_start.to(obs.dtype).reshape(-1, 1))
        base_features = self.actor_body(obs)
        next_hidden = self.actor_gru(base_features, hidden_state)
        features = base_features + torch.tanh(self.temporal_gate) * self.temporal_projection(next_hidden)
        return features, next_hidden

    def distribution_from_features(self, features: torch.Tensor) -> Normal:
        mean = torch.clamp(self.actor_mean(features), -5.0, 5.0)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _atanh(action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    @staticmethod
    def _squash_correction(pre_tanh: torch.Tensor) -> torch.Tensor:
        # Stable log(1 - tanh(x)^2) from the SAC appendix.
        return 2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))

    def _squashed_log_prob(
        self,
        dist: Normal,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        return (
            dist.log_prob(pre_tanh) - self._squash_correction(pre_tanh)
        ).sum(-1)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(
                f"critic observation last dimension must be {self.critic_obs_dim}, "
                f"got {critic_obs.shape[-1]}"
            )
        return self.critic(self.critic_body(critic_obs)).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ):
        features, next_hidden = self._actor_features_step(obs, hidden_state, episode_start)
        dist = self.distribution_from_features(features)
        pre_tanh = dist.rsample()
        action = torch.clamp(
            torch.tanh(pre_tanh),
            -1.0 + self.ACTION_EPS,
            1.0 - self.ACTION_EPS,
        )
        log_prob = self._squashed_log_prob(dist, self._atanh(action))
        value = self.value(critic_obs)
        return action, log_prob, value, next_hidden

    def deterministic_action(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, next_hidden = self._actor_features_step(obs, hidden_state, episode_start)
        dist = self.distribution_from_features(features)
        action = torch.clamp(
            torch.tanh(dist.mean),
            -1.0 + self.ACTION_EPS,
            1.0 - self.ACTION_EPS,
        )
        return action, next_hidden

    def act_deterministic(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ):
        features, next_hidden = self._actor_features_step(
            obs, hidden_state, episode_start
        )
        dist = self.distribution_from_features(features)
        action = torch.clamp(
            torch.tanh(dist.mean),
            -1.0 + self.ACTION_EPS,
            1.0 - self.ACTION_EPS,
        )
        log_prob = self._squashed_log_prob(dist, self._atanh(action))
        return action, log_prob, self.value(critic_obs), next_hidden

    def distribution_parameters_sequence(
        self,
        obs: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim != 3:
            raise ValueError(
                "recurrent observations must have shape (time, batch, obs_dim)"
            )
        hidden = initial_hidden
        means = []
        stds = []
        for step in range(obs.shape[0]):
            features, hidden = self._actor_features_step(
                obs[step], hidden, episode_starts[step]
            )
            dist = self.distribution_from_features(features)
            means.append(dist.mean)
            stds.append(dist.stddev)
        return torch.stack(means), torch.stack(stds), hidden

    def evaluate_actions_sequence(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ):
        if obs.ndim != 3:
            raise ValueError("recurrent PPO observations must have shape (time, batch, obs_dim)")
        time_steps, batch_size, _ = obs.shape
        hidden = initial_hidden
        log_probs = []
        entropies = []
        for step in range(time_steps):
            features, hidden = self._actor_features_step(
                obs[step],
                hidden,
                episode_starts[step],
            )
            dist = self.distribution_from_features(features)
            bounded_actions = torch.clamp(
                actions[step],
                -1.0 + self.ACTION_EPS,
                1.0 - self.ACTION_EPS,
            )
            pre_tanh = self._atanh(bounded_actions)
            log_probs.append(self._squashed_log_prob(dist, pre_tanh))
            entropy_pre_tanh = dist.rsample()
            entropies.append(-self._squashed_log_prob(dist, entropy_pre_tanh))
        flat_critic_obs = critic_obs.reshape(time_steps * batch_size, self.critic_obs_dim)
        values = self.value(flat_critic_obs).reshape(time_steps, batch_size)
        return torch.stack(log_probs), torch.stack(entropies), values, hidden

    def model_config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "actor_hidden_sizes": list(self.actor_hidden_sizes),
            "critic_hidden_sizes": list(self.critic_hidden_sizes),
            "recurrent_hidden_size": self.recurrent_hidden_size,
            "temporal_gate_raw": float(self.temporal_gate.detach().cpu()),
            "temporal_gate_effective": float(
                torch.tanh(self.temporal_gate).detach().cpu()
            ),
            "temporal_gate": float(torch.tanh(self.temporal_gate).detach().cpu()),
            "activation": self.activation,
            "action_distribution": "tanh_squashed_normal",
            "recurrent": True,
            "asymmetric_critic": self.critic_obs_dim != self.obs_dim,
        }


class PopArtValueNormalizer:
    """EMA return normalization with value-output preservation."""

    def __init__(
        self,
        device: torch.device,
        beta: float = 0.999,
        epsilon: float = 1e-5,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("PopArt beta must be in [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("PopArt epsilon must be positive")
        self.device = device
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.mean = torch.zeros((), dtype=torch.float32, device=device)
        self.second_moment = torch.ones((), dtype=torch.float32, device=device)
        self.std = torch.ones((), dtype=torch.float32, device=device)
        self.initialized = False
        self.num_updates = 0

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    @torch.no_grad()
    def update(
        self,
        returns: torch.Tensor,
        output_layer: nn.Linear,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        finite_returns = returns.detach().reshape(-1)
        finite_returns = finite_returns[torch.isfinite(finite_returns)]
        if finite_returns.numel() == 0:
            return
        batch_mean = finite_returns.mean()
        batch_second_moment = torch.square(finite_returns).mean()
        old_mean = self.mean.clone()
        old_std = self.std.clone()
        if self.initialized:
            new_mean = self.beta * self.mean + (1.0 - self.beta) * batch_mean
            new_second_moment = (
                self.beta * self.second_moment
                + (1.0 - self.beta) * batch_second_moment
            )
        else:
            new_mean = batch_mean
            new_second_moment = batch_second_moment
            self.initialized = True
        new_variance = torch.clamp(
            new_second_moment - torch.square(new_mean),
            min=self.epsilon * self.epsilon,
        )
        new_std = torch.sqrt(new_variance)
        scale = old_std / new_std
        output_layer.weight.mul_(scale)
        output_layer.bias.copy_(
            (old_std * output_layer.bias + old_mean - new_mean) / new_std
        )
        if optimizer is not None:
            for parameter in (output_layer.weight, output_layer.bias):
                state = optimizer.state.get(parameter, {})
                if "exp_avg" in state:
                    state["exp_avg"].mul_(scale)
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(scale * scale)
        self.mean.copy_(new_mean)
        self.second_moment.copy_(new_second_moment)
        self.std.copy_(new_std)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "beta": self.beta,
            "epsilon": self.epsilon,
            "mean": self.mean.detach().cpu(),
            "second_moment": self.second_moment.detach().cpu(),
            "std": self.std.detach().cpu(),
            "initialized": self.initialized,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.beta = float(state.get("beta", self.beta))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.mean.copy_(torch.as_tensor(state["mean"], device=self.device))
        self.second_moment.copy_(
            torch.as_tensor(state["second_moment"], device=self.device)
        )
        self.std.copy_(torch.as_tensor(state["std"], device=self.device))
        self.initialized = bool(state.get("initialized", True))
        self.num_updates = int(state.get("num_updates", 0))


def compute_gae_vec(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    steps, num_envs = rewards.shape
    advantages = np.zeros((steps, num_envs), dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(steps)):
        next_values = last_values if t == steps - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns.astype(np.float32), advantages.astype(np.float32)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def append_csv_row(path: Path, fieldnames: Sequence[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def start_terminal_log(run_dir: Path):
    log_dir = run_dir / "terminal_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"
    log_file = log_path.open("a", buffering=1, encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"[FAST PPO] terminal output is being saved to {log_path}")
    return log_file, original_stdout, original_stderr


def start_tensorboard_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[FAST PPO] TensorBoard unavailable ({exc}); continuing without it")
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def write_tensorboard_scalars(writer, metrics: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            writer.add_scalar(key, float(value), step)


def save_training_plots(run_dir: Path, update_rows: List[dict], episode_rows: List[dict]) -> None:
    matplotlib_cache_dir = run_dir / "matplotlib_cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[FAST PPO] plotting unavailable ({exc}); skipping plots")
        return

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    except Exception:
        run_args = {}
    if update_rows:
        x = [r["total_steps"] for r in update_rows]
        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        axes[0].plot(x, [r["mean_rollout_reward"] for r in update_rows])
        axes[0].set_ylabel("mean reward")
        axes[1].plot(x, [r["mean_xy_err"] for r in update_rows], label="xy")
        axes[1].plot(x, [r["mean_z_err"] for r in update_rows], label="z")
        axes[1].legend()
        axes[1].set_ylabel("error")
        axes[2].plot(x, [r["mean_speed_xy"] for r in update_rows], label="speed xy")
        axes[2].plot(x, [r["mean_tracking_velocity_error"] for r in update_rows], label="tracking velocity error")
        axes[2].plot(
            x,
            [r.get("mean_vertical_motion_velocity_error", 0.0) for r in update_rows],
            label="vertical motion velocity error",
        )
        axes[2].legend()
        axes[2].set_ylabel("speed")
        axes[3].plot(x, [r["success_count"] for r in update_rows], label="success")
        axes[3].plot(x, [r["timeout_count"] for r in update_rows], label="timeout")
        axes[3].plot(x, [r["other_done_count"] for r in update_rows], label="other")
        axes[3].legend()
        axes[3].set_ylabel("done count")
        axes[3].set_xlabel("env steps")
        fig.tight_layout()
        fig.savefig(plot_dir / "update_metrics.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        axes[0].plot(x, [r.get("capture_phase_fraction", 0.0) for r in update_rows], label="capture")
        axes[0].plot(x, [r["moving_phase_fraction"] for r in update_rows], label="moving")
        axes[0].plot(x, [r["decelerating_phase_fraction"] for r in update_rows], label="decelerating")
        axes[0].plot(x, [r["stopped_phase_fraction"] for r in update_rows], label="stopped")
        axes[0].set_ylabel("phase fraction")
        axes[0].legend()
        axes[1].plot(x, [r["mean_stopped_xy_err"] for r in update_rows], label="stopped xy error")
        axes[1].set_ylabel("position error")
        axes[1].legend()
        axes[2].plot(x, [r["mean_stopped_speed_xy"] for r in update_rows], label="actual speed")
        axes[2].plot(x, [r["mean_desired_approach_speed"] for r in update_rows], label="desired speed")
        axes[2].plot(x, [r["mean_allowed_stopped_speed"] for r in update_rows], label="allowed speed")
        axes[2].plot(x, [r["mean_stopped_velocity_error"] for r in update_rows], label="velocity error")
        axes[2].set_ylabel("stopped velocity")
        axes[2].legend()
        axes[3].plot(x, [r["goal_zone_fraction"] for r in update_rows], label="goal zone")
        axes[3].plot(x, [r["mean_goal_dwell_fraction"] for r in update_rows], label="mean dwell")
        axes[3].plot(
            x,
            [r.get("capture_acquired_fraction", 0.0) for r in update_rows],
            label="capture acquired",
        )
        axes[3].plot(x, [r["stopped_xy_zone_fraction"] for r in update_rows], label="stopped xy")
        axes[3].plot(x, [r["stopped_z_zone_fraction"] for r in update_rows], label="stopped z")
        axes[3].plot(x, [r["stopped_position_zone_fraction"] for r in update_rows], label="stopped position")
        axes[3].plot(x, [r["stopped_stationary_fraction"] for r in update_rows], label="stopped stationary")
        axes[3].set_ylabel("fraction")
        axes[3].set_xlabel("env steps")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "phase_and_stopping.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        axes[0].plot(
            x,
            [r.get("moving_xy_good_sample_fraction", 0.0) for r in update_rows],
            label="XY",
        )
        axes[0].plot(
            x,
            [r.get("moving_z_good_sample_fraction", 0.0) for r in update_rows],
            label="Z",
        )
        axes[0].plot(
            x,
            [r.get("moving_velocity_good_sample_fraction", 0.0) for r in update_rows],
            label="velocity",
        )
        axes[0].plot(
            x,
            [r.get("moving_good_sample_fraction", 0.0) for r in update_rows],
            label="joint",
        )
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel("moving fraction")
        axes[0].legend()
        axes[1].plot(
            x,
            [r.get("vertical_position_good_sample_fraction", 0.0) for r in update_rows],
            label="position",
        )
        axes[1].plot(
            x,
            [r.get("vertical_velocity_good_sample_fraction", 0.0) for r in update_rows],
            label="vertical velocity",
        )
        axes[1].plot(
            x,
            [r.get("vertical_good_sample_fraction", 0.0) for r in update_rows],
            label="joint",
        )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("vertical fraction")
        axes[1].legend()
        axes[2].plot(
            x,
            [r.get("stopped_xy_good_sample_fraction", 0.0) for r in update_rows],
            label="XY",
        )
        axes[2].plot(
            x,
            [r.get("stopped_z_good_sample_fraction", 0.0) for r in update_rows],
            label="Z",
        )
        axes[2].plot(
            x,
            [r.get("stopped_speed_good_sample_fraction", 0.0) for r in update_rows],
            label="speed",
        )
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("stopped fraction")
        axes[2].legend()
        axes[3].plot(
            x,
            [r.get("temporal_gate_raw", 0.0) for r in update_rows],
            label="raw gate",
        )
        axes[3].plot(
            x,
            [r.get("temporal_gate_effective", 0.0) for r in update_rows],
            label="tanh(gate)",
        )
        axes[3].set_ylabel("GRU residual gate")
        axes[3].set_xlabel("env steps")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "condition_diagnostics.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(x, [r["policy_loss"] for r in update_rows], label="policy loss")
        axes[0].plot(x, [r["value_loss"] for r in update_rows], label="value loss")
        axes[0].legend()
        axes[1].plot(x, [r["approx_kl"] for r in update_rows], label="approx KL")
        axes[1].plot(x, [r["clip_fraction"] for r in update_rows], label="clip fraction")
        axes[1].plot(x, [r["explained_variance"] for r in update_rows], label="explained variance")
        axes[1].legend()
        axes[2].plot(x, [r["action_mean"] for r in update_rows], label="action mean")
        axes[2].plot(x, [r["action_std"] for r in update_rows], label="action std")
        axes[2].plot(x, [r["action_abs_mean"] for r in update_rows], label="action abs mean")
        axes[2].set_xlabel("env steps")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "ppo_diagnostics.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for key, label in (
            ("mean_local_tracking_soft_joint_quality", "soft joint"),
            ("mean_local_tracking_xy_good_fraction", "XY"),
            ("mean_local_tracking_velocity_good_fraction", "velocity"),
            ("mean_local_tracking_z_good_fraction", "Z"),
        ):
            axes[0].plot(x, [r.get(key, 0.0) for r in update_rows], label=label)
        axes[0].set_ylabel("local quality")
        axes[0].legend()
        axes[1].plot(
            x,
            [r.get("mean_local_tracking_xy_drift_delta_m", 0.0) for r in update_rows],
            label="XY drift",
        )
        axes[1].plot(
            x,
            [r.get("mean_reward_local_tracking", 0.0) for r in update_rows],
            label="tracking reward",
        )
        axes[1].plot(
            x,
            [r.get("mean_reward_local_drift", 0.0) for r in update_rows],
            label="drift reward",
        )
        axes[1].set_ylabel("local credit")
        axes[1].legend()
        axes[2].plot(
            x,
            [r.get("value_clip_fraction", 0.0) for r in update_rows],
            label="value clip fraction",
        )
        axes[2].plot(
            x,
            [r.get("reference_kl", 0.0) for r in update_rows],
            label="reference KL",
        )
        axes[2].plot(
            x,
            [r.get("cmd_saturation_fraction", 0.0) for r in update_rows],
            label="CTBR saturation",
        )
        axes[2].set_xlabel("env steps")
        axes[2].set_ylabel("optimizer/control")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "local_credit_and_control.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        axes[0].plot(
            x,
            [r.get("checkpoint_score", 0.0) for r in update_rows],
            label="rolling score",
        )
        axes[0].plot(
            x,
            [r.get("best_checkpoint_score", 0.0) for r in update_rows],
            label="best score",
        )
        axes[0].set_ylabel("checkpoint score")
        axes[0].legend()
        axes[1].plot(
            x,
            [r.get("overall_success_rate", 0.0) for r in update_rows],
            label="overall success",
        )
        axes[1].plot(
            x,
            [r.get("timeout_rate", 0.0) for r in update_rows],
            label="timeout",
        )
        axes[1].plot(
            x,
            [r.get("other_failure_rate", 0.0) for r in update_rows],
            label="other failure",
        )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("episode outcome")
        axes[1].legend()
        for key, label in (
            ("stopped_xy_zone_fraction", "stopped XY"),
            ("stopped_position_zone_fraction", "stopped position"),
            ("stopped_stationary_fraction", "stopped stationary"),
            ("final_stop_good_fraction", "final-stop joint"),
        ):
            axes[2].plot(x, [r.get(key, 0.0) for r in update_rows], label=label)
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("final-stop quality")
        axes[2].legend()
        axes[3].plot(
            x,
            [r.get("mean_completed_final_xy_err", 0.0) for r in update_rows],
            label="completed final XY error",
        )
        axes[3].axhline(
            float(run_args.get("tracking_xy_tolerance_m", 0.3)),
            color="tab:red",
            linestyle="--",
            label="final XY tolerance",
        )
        axes[3].set_ylabel("final XY error (m)")
        axes[3].set_xlabel("env steps")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "checkpoint_and_outcomes.png", dpi=150)
        plt.close(fig)

        if bool(run_args.get("camera_tracking_enabled", False)):
            fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
            for key, label in (
                ("camera_visible_sample_fraction", "visible"),
                ("camera_good_sample_fraction", "success FOV"),
                ("camera_success_met_fraction", "episode gate met"),
            ):
                axes[0].plot(
                    x, [r.get(key, 0.0) for r in update_rows], label=label
                )
            axes[0].axhline(
                float(run_args.get("camera_success_min_fraction", 0.9)),
                color="tab:red",
                linestyle="--",
                label="success threshold",
            )
            axes[0].set_ylim(0.0, 1.05)
            axes[0].set_ylabel("camera fraction")
            axes[0].legend()
            axes[1].plot(
                x,
                [r.get("mean_camera_center_quality", 0.0) for r in update_rows],
                label="center quality",
            )
            axes[1].plot(
                x,
                [r.get("mean_abs_camera_bearing_rad", 0.0) for r in update_rows],
                label="abs bearing (rad)",
            )
            axes[1].plot(
                x,
                [r.get("mean_abs_camera_elevation_rad", 0.0) for r in update_rows],
                label="abs elevation (rad)",
            )
            axes[1].set_ylabel("image geometry")
            axes[1].legend()
            axes[2].plot(
                x,
                [r.get("mean_camera_local_good_fraction", 0.0) for r in update_rows],
                label="local success FOV",
            )
            axes[2].plot(
                x,
                [r.get("mean_camera_local_center_quality", 0.0) for r in update_rows],
                label="local center quality",
            )
            axes[2].plot(
                x,
                [r.get("mean_camera_max_consecutive_lost_sec", 0.0) for r in update_rows],
                label="max lost streak (s)",
            )
            axes[2].set_ylabel("local camera")
            axes[2].legend()
            for key, label in (
                ("mean_reward_camera_center", "center"),
                ("mean_reward_camera_visible", "visible"),
                ("mean_reward_camera_lost", "lost"),
                ("mean_reward_camera_local", "joint 2s"),
            ):
                axes[3].plot(
                    x, [r.get(key, 0.0) for r in update_rows], label=label
                )
            axes[3].set_ylabel("camera reward")
            axes[3].set_xlabel("env steps")
            axes[3].legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "camera_tracking.png", dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(x, [r["mean_reward_progress"] for r in update_rows], label="progress")
        axes[0].plot(x, [r["mean_reward_distance"] for r in update_rows], label="distance")
        axes[0].plot(
            x,
            [r.get("mean_reward_moving_position", 0.0) for r in update_rows],
            label="moving position",
        )
        axes[0].plot(
            x,
            [r.get("mean_reward_moving_progress", 0.0) for r in update_rows],
            label="moving progress",
        )
        axes[0].plot(
            x,
            [r.get("mean_reward_moving_recovery", 0.0) for r in update_rows],
            label="moving recovery",
        )
        axes[0].plot(
            x,
            [r.get("mean_reward_stopped_position", 0.0) for r in update_rows],
            label="stopped position",
        )
        axes[0].plot(x, [r["mean_reward_z"] for r in update_rows], label="z")
        axes[0].legend()
        axes[1].plot(x, [r["mean_reward_speed"] for r in update_rows], label="speed")
        axes[1].plot(
            x,
            [r.get("mean_reward_moving_velocity", 0.0) for r in update_rows],
            label="moving velocity",
        )
        axes[1].plot(x, [r["mean_reward_braking"] for r in update_rows], label="braking")
        axes[1].plot(
            x,
            [r["mean_reward_stop_overspeed"] for r in update_rows],
            label="stop overspeed",
        )
        axes[1].plot(
            x,
            [r.get("mean_reward_moving_good", 0.0) for r in update_rows],
            label="moving good",
        )
        axes[1].plot(
            x,
            [r.get("mean_reward_local_tracking", 0.0) for r in update_rows],
            label="local tracking",
        )
        axes[1].plot(
            x,
            [r.get("mean_reward_local_drift", 0.0) for r in update_rows],
            label="local drift",
        )
        axes[1].legend()
        axes[2].plot(x, [r.get("mean_reward_capture", 0.0) for r in update_rows], label="capture")
        axes[2].plot(x, [r.get("mean_reward_time", 0.0) for r in update_rows], label="stop time")
        axes[2].plot(x, [r["mean_reward_goal_zone"] for r in update_rows], label="goal zone")
        axes[2].plot(x, [r["mean_reward_dwell"] for r in update_rows], label="dwell")
        axes[2].plot(x, [r["mean_reward_success"] for r in update_rows], label="success")
        axes[2].plot(x, [r["mean_reward_crash"] for r in update_rows], label="crash")
        axes[2].plot(x, [r.get("mean_reward_timeout", 0.0) for r in update_rows], label="timeout")
        axes[2].set_xlabel("env steps")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "reward_components.png", dpi=150)
        plt.close(fig)

        primitive_metrics = (
            ("primitive_mean_xy_err", "mean XY error"),
            ("primitive_mean_z_err", "mean Z error"),
            ("primitive_mean_velocity_error", "mean velocity error"),
            (
                "primitive_mean_vertical_velocity_error",
                "mean vertical velocity error",
            ),
            ("primitive_good_fraction", "good fraction"),
        )
        parsed_metrics = {
            metric: [
                json.loads(row.get(metric, "{}"))
                if isinstance(row.get(metric, "{}"), str)
                else row.get(metric, {})
                for row in update_rows
            ]
            for metric, _ in primitive_metrics
        }
        primitive_names = sorted(
            {
                primitive
                for values in parsed_metrics.values()
                for row in values
                for primitive in row
                if primitive not in ("capture", "stopped")
            }
        )
        if primitive_names:
            fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
            for axis, (metric, ylabel) in zip(axes, primitive_metrics):
                rows = parsed_metrics[metric]
                for primitive in primitive_names:
                    axis.plot(
                        x,
                        [float(row.get(primitive, np.nan)) for row in rows],
                        label=primitive,
                    )
                axis.set_ylabel(ylabel)
                axis.legend()
            axes[-1].set_xlabel("env steps")
            fig.tight_layout()
            fig.savefig(plot_dir / "primitive_performance.png", dpi=150)
            plt.close(fig)

            condition_metrics = (
                ("primitive_xy_good_fraction", "XY good fraction"),
                ("primitive_z_good_fraction", "Z good fraction"),
                ("primitive_velocity_good_fraction", "velocity good fraction"),
                (
                    "primitive_vertical_velocity_good_fraction",
                    "vertical velocity good fraction",
                ),
                ("primitive_good_fraction", "joint good fraction"),
            )
            parsed_conditions = {
                metric: [
                    json.loads(row.get(metric, "{}"))
                    if isinstance(row.get(metric, "{}"), str)
                    else row.get(metric, {})
                    for row in update_rows
                ]
                for metric, _ in condition_metrics
            }
            fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
            for axis, (metric, ylabel) in zip(axes, condition_metrics):
                rows = parsed_conditions[metric]
                for primitive in primitive_names:
                    axis.plot(
                        x,
                        [float(row.get(primitive, np.nan)) for row in rows],
                        label=primitive,
                    )
                axis.set_ylim(0.0, 1.05)
                axis.set_ylabel(ylabel)
                axis.legend()
            axes[-1].set_xlabel("env steps")
            fig.tight_layout()
            fig.savefig(plot_dir / "primitive_conditions.png", dpi=150)
            plt.close(fig)
    if episode_rows:
        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        x = [r["total_steps"] for r in episode_rows]
        axes[0].plot(x, [r["return"] for r in episode_rows], ".", markersize=3)
        axes[0].set_ylabel("episode return")
        axes[1].plot(x, [r["final_goal_xy_err"] for r in episode_rows], ".", markersize=3, label="xy")
        axes[1].plot(x, [r["final_z_err"] for r in episode_rows], ".", markersize=3, label="z")
        axes[1].legend()
        axes[1].set_ylabel("final error")
        axes[2].plot(x, [r["final_speed_xy"] for r in episode_rows], ".", markersize=3, label="final speed xy")
        axes[2].plot(
            x,
            [r["mean_tracking_velocity_error"] for r in episode_rows],
            ".",
            markersize=3,
            label="mean velocity error",
        )
        axes[2].legend()
        axes[2].set_ylabel("speed")
        success_flags = np.asarray(
            [1.0 if r["success"] else 0.0 for r in episode_rows],
            dtype=np.float64,
        )
        cumulative_success = np.cumsum(success_flags) / np.arange(1, len(success_flags) + 1)
        axes[3].plot(x, cumulative_success, label="cumulative success rate")
        axes[3].set_ylabel("success rate")
        axes[3].set_xlabel("env steps")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "episode_metrics.png", dpi=150)
        plt.close(fig)

        xy_tolerance = float(run_args.get("tracking_xy_tolerance_m", 0.3))
        z_tolerance = float(run_args.get("tracking_z_tolerance_m", 0.3))
        speed_tolerance = float(run_args.get("stopped_speed_xy_tolerance_mps", 0.25))
        moving_good_tolerance = float(run_args.get("moving_success_min_fraction", 0.5))
        vertical_good_tolerance = float(
            run_args.get("vertical_success_min_fraction", 0.75)
        )
        episode_index = np.arange(1, len(episode_rows) + 1)
        colors = ["tab:green" if r["success"] else "tab:red" for r in episode_rows]
        cumulative_success = np.cumsum(success_flags) / episode_index
        camera_enabled = bool(run_args.get("camera_tracking_enabled", False))
        outcome_panel_count = 5 if camera_enabled else 4

        fig, axes = plt.subplots(
            outcome_panel_count,
            1,
            figsize=(12, 20 if camera_enabled else 17),
            sharex=True,
        )
        final_xy = [r["final_goal_xy_err"] for r in episode_rows]
        axes[0].plot(episode_index, final_xy, color="tab:blue", alpha=0.18)
        axes[0].scatter(episode_index, final_xy, c=colors, s=18, label="final goal xy error")
        axes[0].axhline(xy_tolerance, color="tab:red", linestyle="--", label="xy tolerance")
        axes[0].set_ylabel("final goal xy error (m)")
        axes[0].legend(loc="upper left")
        success_axis = axes[0].twinx()
        success_axis.plot(
            episode_index,
            cumulative_success,
            color="tab:green",
            label="cumulative success rate",
        )
        success_axis.set_ylim(0.0, 1.05)
        success_axis.set_ylabel("success rate")
        success_axis.legend(loc="lower right")

        final_z = [r["final_z_err"] for r in episode_rows]
        axes[1].plot(episode_index, final_z, color="tab:purple", alpha=0.18)
        axes[1].scatter(episode_index, final_z, c=colors, s=18, label="final z error")
        axes[1].axhline(z_tolerance, color="tab:red", linestyle="--", label="z tolerance")
        axes[1].set_ylabel("final z error (m)")
        axes[1].legend(loc="upper left")

        moving_good = [r.get("moving_good_fraction", 0.0) for r in episode_rows]
        axes[2].plot(episode_index, moving_good, color="tab:cyan", alpha=0.25)
        axes[2].scatter(
            episode_index,
            moving_good,
            c=colors,
            s=18,
            label="moving good fraction",
        )
        axes[2].axhline(
            moving_good_tolerance,
            color="tab:red",
            linestyle="--",
            label="moving success threshold",
        )
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("moving good fraction")
        axes[2].legend(loc="upper left")

        final_speed = [r["final_speed_xy"] for r in episode_rows]
        axes[3].plot(episode_index, final_speed, color="tab:orange", alpha=0.65, label="final speed xy")
        axes[3].axhline(
            speed_tolerance,
            color="tab:red",
            linestyle="--",
            label="stopped speed tolerance",
        )
        axes[3].set_ylabel("final speed xy (m/s)")
        axes[3].legend(loc="upper left")

        if camera_enabled:
            camera_good = [
                r.get("camera_good_fraction", 0.0) for r in episode_rows
            ]
            axes[4].plot(
                episode_index,
                [r.get("camera_visible_fraction", 0.0) for r in episode_rows],
                color="tab:blue",
                alpha=0.45,
                label="camera visible fraction",
            )
            axes[4].plot(
                episode_index,
                camera_good,
                color="tab:cyan",
                alpha=0.25,
            )
            axes[4].scatter(
                episode_index,
                camera_good,
                c=colors,
                s=18,
                label="camera success-region fraction",
            )
            axes[4].axhline(
                float(run_args.get("camera_success_min_fraction", 0.9)),
                color="tab:red",
                linestyle="--",
                label="camera success threshold",
            )
            axes[4].set_ylim(0.0, 1.05)
            axes[4].set_ylabel("camera / heading fraction")
            axes[4].legend(loc="lower left")
        axes[-1].set_xlabel("completed episode")
        fig.tight_layout()
        fig.savefig(plot_dir / "episode_outcomes.png", dpi=150)
        plt.close(fig)

        condition_count = 4 if camera_enabled else 3
        fig, axes = plt.subplots(
            condition_count,
            1,
            figsize=(12, 14 if camera_enabled else 11),
            sharex=True,
        )
        for key, label in (
            ("moving_xy_good_fraction", "XY"),
            ("moving_z_good_fraction", "Z"),
            ("moving_velocity_good_fraction", "velocity"),
            ("moving_good_fraction", "joint"),
        ):
            axes[0].plot(
                episode_index,
                [r.get(key, 0.0) for r in episode_rows],
                ".",
                markersize=3,
                label=label,
            )
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel("moving fraction")
        axes[0].legend()
        for key, label in (
            ("vertical_position_good_fraction", "position"),
            ("vertical_velocity_good_fraction", "vertical velocity"),
            ("vertical_good_fraction", "joint"),
        ):
            axes[1].plot(
                episode_index,
                [r.get(key, 0.0) for r in episode_rows],
                ".",
                markersize=3,
                label=label,
            )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].axhline(
            vertical_good_tolerance,
            color="tab:red",
            linestyle="--",
            label="vertical success threshold",
        )
        axes[1].set_ylabel("vertical fraction")
        axes[1].legend()
        for key, label in (
            ("stopped_xy_good_fraction", "XY"),
            ("stopped_z_good_fraction", "Z"),
            ("stopped_speed_good_fraction", "speed"),
        ):
            axes[2].plot(
                episode_index,
                [r.get(key, 0.0) for r in episode_rows],
                ".",
                markersize=3,
                label=label,
            )
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("stopped fraction")
        axes[2].legend()
        if camera_enabled:
            axes[3].plot(
                episode_index,
                [r.get("camera_visible_fraction", 0.0) for r in episode_rows],
                ".",
                markersize=3,
                label="visible",
            )
            axes[3].plot(
                episode_index,
                [r.get("camera_good_fraction", 0.0) for r in episode_rows],
                ".",
                markersize=3,
                label="success FOV",
            )
            axes[3].axhline(
                float(run_args.get("camera_success_min_fraction", 0.9)),
                color="tab:red",
                linestyle="--",
                label="camera success threshold",
            )
            axes[3].set_ylim(0.0, 1.05)
            axes[3].set_ylabel("camera fraction")
            axes[3].legend()
        axes[-1].set_xlabel("completed episode")
        fig.tight_layout()
        fig.savefig(plot_dir / "episode_condition_fractions.png", dpi=150)
        plt.close(fig)

        def save_conditioned_performance_plot(
            rows: List[dict],
            condition_key: str,
            condition_min: float,
            condition_max: float,
            num_bins: int,
            x_label: str,
            output_name: str,
        ) -> None:
            condition_values = np.asarray(
                [float(row[condition_key]) for row in rows],
                dtype=np.float64,
            )
            condition_success = np.asarray(
                [1.0 if row["success"] else 0.0 for row in rows],
                dtype=np.float64,
            )
            condition_moving_good = np.asarray(
                [float(row.get("moving_good_fraction", 0.0)) for row in rows],
                dtype=np.float64,
            )
            condition_xy_error = np.asarray(
                [float(row.get("mean_goal_xy_err", 0.0)) for row in rows],
                dtype=np.float64,
            )
            condition_z_error = np.asarray(
                [float(row.get("mean_z_err", 0.0)) for row in rows],
                dtype=np.float64,
            )

            value_min = min(float(condition_min), float(condition_max))
            value_max = max(float(condition_min), float(condition_max))
            if value_max - value_min < 1e-6:
                value_min -= 0.5
                value_max += 0.5
            bin_edges = np.linspace(value_min, value_max, max(1, int(num_bins)) + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = float(bin_edges[1] - bin_edges[0])
            bin_ids = np.digitize(condition_values, bin_edges[1:-1], right=False)

            counts = np.zeros(len(bin_centers), dtype=np.int64)
            binned_success = np.full(len(bin_centers), np.nan, dtype=np.float64)
            binned_moving_good = np.full(len(bin_centers), np.nan, dtype=np.float64)
            binned_xy_error = np.full(len(bin_centers), np.nan, dtype=np.float64)
            binned_z_error = np.full(len(bin_centers), np.nan, dtype=np.float64)
            for bin_id in range(len(bin_centers)):
                mask = bin_ids == bin_id
                counts[bin_id] = int(np.count_nonzero(mask))
                if counts[bin_id] == 0:
                    continue
                binned_success[bin_id] = float(np.mean(condition_success[mask]))
                binned_moving_good[bin_id] = float(np.mean(condition_moving_good[mask]))
                binned_xy_error[bin_id] = float(np.mean(condition_xy_error[mask]))
                binned_z_error[bin_id] = float(np.mean(condition_z_error[mask]))

            condition_colors = [
                "tab:green" if row["success"] else "tab:red" for row in rows
            ]
            fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
            axes[0].scatter(
                condition_values,
                condition_moving_good,
                c=condition_colors,
                s=14,
                alpha=0.35,
                label="episode moving good fraction",
            )
            axes[0].axhline(
                moving_good_tolerance,
                color="tab:red",
                linestyle="--",
                label="moving success threshold",
            )
            axes[0].set_ylim(0.0, 1.05)
            axes[0].set_ylabel("moving good fraction")
            axes[0].legend(loc="lower left")

            axes[1].plot(
                bin_centers,
                binned_success,
                marker="o",
                color="tab:green",
                label="binned success rate",
            )
            axes[1].plot(
                bin_centers,
                binned_moving_good,
                marker="o",
                color="tab:cyan",
                label="binned moving good fraction",
            )
            axes[1].axhline(
                moving_good_tolerance,
                color="tab:red",
                linestyle="--",
                label="moving success threshold",
            )
            axes[1].set_ylim(0.0, 1.05)
            axes[1].set_ylabel("fraction")
            axes[1].legend(loc="lower left")

            axes[2].plot(
                bin_centers,
                binned_xy_error,
                marker="o",
                color="tab:blue",
                label="binned mean xy error",
            )
            axes[2].plot(
                bin_centers,
                binned_z_error,
                marker="o",
                color="tab:purple",
                label="binned mean z error",
            )
            axes[2].set_ylabel("mean episode error (m)")
            axes[2].set_xlabel(x_label)
            axes[2].legend(loc="upper left")
            count_axis = axes[2].twinx()
            count_axis.bar(
                bin_centers,
                counts,
                width=0.8 * bin_width,
                color="tab:gray",
                alpha=0.15,
                label="episode count",
            )
            count_axis.set_ylabel("episode count")
            count_axis.legend(loc="upper right")

            axes[2].set_xlim(value_min, value_max)
            for axis in axes:
                axis.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(plot_dir / output_name, dpi=150)
            plt.close(fig)

        direction_rows = [
            row for row in episode_rows if row.get("line_yaw_deg") is not None
        ]
        if direction_rows and bool(run_args.get("randomize_line_yaw", False)):
            angle_min = float(run_args.get("line_yaw_min_deg", -180.0))
            angle_max = float(run_args.get("line_yaw_max_deg", 180.0))
            angle_span = abs(angle_max - angle_min)
            num_angle_bins = max(4, min(36, int(np.ceil(angle_span / 20.0))))
            save_conditioned_performance_plot(
                direction_rows,
                "line_yaw_deg",
                angle_min,
                angle_max,
                num_angle_bins,
                "line yaw (deg)",
                "direction_performance.png",
            )

        radius_rows = [
            row
            for row in episode_rows
            if row.get("sampled_line_length_m") is not None
        ]
        if radius_rows and bool(run_args.get("randomize_line_length", False)):
            radius_min = float(run_args.get("line_length_min_m", 0.0))
            radius_max = float(run_args.get("line_length_max_m", 1.0))
            radius_span = abs(radius_max - radius_min)
            num_radius_bins = max(5, min(20, int(np.ceil(radius_span / 0.5))))
            save_conditioned_performance_plot(
                radius_rows,
                "sampled_line_length_m",
                radius_min,
                radius_max,
                num_radius_bins,
                "desired endpoint radius (m)",
                "radius_performance.png",
            )

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(x, [r["return_progress"] for r in episode_rows], label="progress")
        axes[0].plot(x, [r["return_distance"] for r in episode_rows], label="distance")
        axes[0].plot(
            x,
            [r.get("return_moving_position", 0.0) for r in episode_rows],
            label="moving position",
        )
        axes[0].plot(
            x,
            [r.get("return_moving_recovery", 0.0) for r in episode_rows],
            label="moving recovery",
        )
        axes[0].plot(
            x,
            [r.get("return_stopped_position", 0.0) for r in episode_rows],
            label="stopped position",
        )
        axes[0].plot(x, [r["return_speed"] for r in episode_rows], label="speed")
        axes[0].plot(
            x,
            [r.get("return_moving_velocity", 0.0) for r in episode_rows],
            label="moving velocity",
        )
        axes[0].plot(x, [r["return_braking"] for r in episode_rows], label="braking")
        axes[0].plot(x, [r.get("return_moving_good", 0.0) for r in episode_rows], label="moving good")
        axes[0].plot(
            x,
            [r.get("return_local_tracking", 0.0) for r in episode_rows],
            label="local tracking",
        )
        axes[0].plot(
            x,
            [r.get("return_local_drift", 0.0) for r in episode_rows],
            label="local drift",
        )
        axes[0].legend()
        axes[1].plot(
            x,
            [r["return_stop_overspeed"] for r in episode_rows],
            label="stop overspeed",
        )
        axes[1].plot(x, [r.get("return_capture", 0.0) for r in episode_rows], label="capture")
        axes[1].plot(x, [r.get("return_time", 0.0) for r in episode_rows], label="stop time")
        axes[1].plot(
            x,
            [r.get("return_camera_center", 0.0) for r in episode_rows],
            label="camera center",
        )
        axes[1].plot(
            x,
            [r.get("return_camera_visible", 0.0) for r in episode_rows],
            label="camera visible",
        )
        axes[1].plot(
            x,
            [r.get("return_camera_lost", 0.0) for r in episode_rows],
            label="camera lost",
        )
        axes[1].plot(
            x,
            [r.get("return_camera_local", 0.0) for r in episode_rows],
            label="camera local",
        )
        axes[1].plot(x, [r["return_goal_zone"] for r in episode_rows], label="goal zone")
        axes[1].plot(x, [r["return_dwell"] for r in episode_rows], label="dwell")
        axes[1].plot(x, [r["return_success"] for r in episode_rows], label="success")
        axes[1].plot(x, [r["return_crash"] for r in episode_rows], label="crash")
        axes[1].plot(x, [r.get("return_timeout", 0.0) for r in episode_rows], label="timeout")
        axes[1].set_xlabel("env steps")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "episode_reward_components.png", dpi=150)
        plt.close(fig)


def load_transplanted_checkpoint(
    policy: ActorCritic,
    checkpoint: Path,
    device: torch.device,
    allow_partial: bool = False,
) -> dict:
    # Keep optimizer and RNG payloads on CPU. Model tensors are moved to the
    # policy device individually below, while CUDA RNG restoration requires
    # CPU ByteTensors.
    try:
        state_dict = torch.load(
            checkpoint,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    checkpoint_payload = state_dict
    if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint at {checkpoint} is not a state_dict")

    current = policy.state_dict()
    merged = {key: value.clone() for key, value in current.items()}
    exact = []
    partial = []
    skipped = []
    for key, src_value in state_dict.items():
        if key not in merged:
            skipped.append(f"{key}: not in target policy")
            continue
        dst_value = merged[key]
        src_tensor = src_value.detach().to(device=dst_value.device, dtype=dst_value.dtype)
        if tuple(src_tensor.shape) == tuple(dst_value.shape):
            merged[key] = src_tensor.clone()
            exact.append(key)
            continue
        if allow_partial and src_tensor.ndim == dst_value.ndim and src_tensor.ndim > 0:
            slices = tuple(slice(0, min(src_tensor.shape[i], dst_value.shape[i])) for i in range(src_tensor.ndim))
            copied = dst_value.clone()
            # Appended observation columns must start inert. This preserves the
            # exact transferred policy/value function until gradients learn how
            # to use the new sensors.
            if (
                src_tensor.ndim >= 2
                and src_tensor.shape[:-1] == dst_value.shape[:-1]
                and src_tensor.shape[-1] < dst_value.shape[-1]
            ):
                copied[..., src_tensor.shape[-1] :] = 0.0
            copied[slices] = src_tensor[slices]
            merged[key] = copied
            partial.append(f"{key}: {tuple(src_tensor.shape)} -> {tuple(dst_value.shape)}")
            continue
        skipped.append(f"{key}: {tuple(src_tensor.shape)} -> {tuple(dst_value.shape)}")
    missing = [key for key in current if key not in state_dict]
    if (partial or skipped or missing) and not allow_partial:
        details = [*skipped, *[f"{key}: missing from checkpoint" for key in missing]]
        for key, src_value in state_dict.items():
            if key in current and tuple(src_value.shape) != tuple(current[key].shape):
                details.append(
                    f"{key}: {tuple(src_value.shape)} -> {tuple(current[key].shape)}"
                )
        preview = "\n  ".join(details[:12])
        raise ValueError(
            "checkpoint architecture/observation shape does not match the current policy. "
            "Use a checkpoint produced by this network, start with --from_scratch, or pass "
            f"--allow_partial_checkpoint explicitly.\n  {preview}"
        )
    policy.load_state_dict(merged)
    load_kind = "partially transplanted" if partial or skipped or missing else "loaded"
    print(f"[FAST PPO] {load_kind} checkpoint from {checkpoint}")
    print(f"[FAST PPO] exact tensors: {len(exact)}, partial tensors: {len(partial)}, skipped: {len(skipped)}")
    if partial:
        print("[FAST PPO] partial tensors:")
        for item in partial[:12]:
            print(f"  {item}")
    if skipped:
        print("[FAST PPO] skipped tensors:")
        for item in skipped[:12]:
            print(f"  {item}")
    return checkpoint_payload if isinstance(checkpoint_payload, dict) else {}


def restore_training_state(
    checkpoint_payload: dict,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    env_rng: np.random.Generator | None = None,
    restore_rng: bool = True,
    value_normalizer: PopArtValueNormalizer | None = None,
) -> bool:
    normalizer_state = checkpoint_payload.get("value_normalizer_state")
    if value_normalizer is not None and normalizer_state is not None:
        value_normalizer.load_state_dict(normalizer_state)
        print("[FAST PPO] restored PopArt value-normalizer state")
    elif value_normalizer is not None:
        print("[FAST PPO] checkpoint has no PopArt state; using identity initialization")
    actor_state = checkpoint_payload.get("actor_optimizer_state")
    critic_state = checkpoint_payload.get("critic_optimizer_state")
    if actor_state is None or critic_state is None:
        print("[FAST PPO] checkpoint has no optimizer state; optimizers start fresh")
        return False
    try:
        actor_optimizer.load_state_dict(actor_state)
        critic_optimizer.load_state_dict(critic_state)
    except (ValueError, KeyError) as exc:
        print(f"[FAST PPO] optimizer state is incompatible ({exc}); optimizers start fresh")
        return False
    if not restore_rng:
        print("[FAST PPO] restored actor/critic optimizer states; kept configured RNG seed")
        return True
    rng_restored = True
    try:
        if "python_random_state" in checkpoint_payload:
            random.setstate(checkpoint_payload["python_random_state"])
        if "numpy_random_state" in checkpoint_payload:
            np.random.set_state(checkpoint_payload["numpy_random_state"])
        if "torch_random_state" in checkpoint_payload:
            torch.set_rng_state(checkpoint_payload["torch_random_state"].cpu())
        cuda_rng_state = checkpoint_payload.get("torch_cuda_random_state")
        if torch.cuda.is_available() and cuda_rng_state is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state])
        if env_rng is not None and checkpoint_payload.get("env_rng_state") is not None:
            env_rng.bit_generator.state = checkpoint_payload["env_rng_state"]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        rng_restored = False
        print(
            f"[FAST PPO] RNG state is incompatible ({exc}); "
            "continuing with current RNG states"
        )
    if rng_restored:
        print("[FAST PPO] restored actor/critic optimizer and RNG states")
    else:
        print("[FAST PPO] restored actor/critic optimizer states")
    return True


def _policy_checkpoint_payload(
    policy: ActorCritic,
    update: int,
    total_steps: int,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    env_rng: np.random.Generator | None = None,
    value_normalizer: PopArtValueNormalizer | None = None,
    reference_policy: ActorCritic | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 5,
        "state_dict": policy.state_dict(),
        "model_config": policy.model_config(),
        "update": int(update),
        "total_steps": int(total_steps),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "env_rng_state": env_rng.bit_generator.state if env_rng is not None else None,
        "value_normalizer_state": (
            value_normalizer.state_dict() if value_normalizer is not None else None
        ),
        "reference_policy_state_dict": (
            reference_policy.state_dict() if reference_policy is not None else None
        ),
    }
    if actor_optimizer is not None:
        payload["actor_optimizer_state"] = actor_optimizer.state_dict()
    if critic_optimizer is not None:
        payload["critic_optimizer_state"] = critic_optimizer.state_dict()
    return payload


def _freeze_checkpoint_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {
            _freeze_checkpoint_value(key): _freeze_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_freeze_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_checkpoint_value(item) for item in value)
    return copy.deepcopy(value)


def capture_policy_checkpoint(
    policy: ActorCritic,
    update: int,
    total_steps: int,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    env_rng: np.random.Generator | None = None,
    value_normalizer: PopArtValueNormalizer | None = None,
    reference_policy: ActorCritic | None = None,
) -> dict[str, Any]:
    """Freeze the exact policy and training state used to collect a rollout."""
    return _freeze_checkpoint_value(
        _policy_checkpoint_payload(
            policy,
            update,
            total_steps,
            actor_optimizer,
            critic_optimizer,
            env_rng,
            value_normalizer,
            reference_policy,
        )
    )


def save_checkpoint_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_policy_checkpoint(
    path: Path,
    policy: ActorCritic,
    update: int,
    total_steps: int,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    env_rng: np.random.Generator | None = None,
    value_normalizer: PopArtValueNormalizer | None = None,
    reference_policy: ActorCritic | None = None,
) -> None:
    save_checkpoint_payload(
        path,
        _policy_checkpoint_payload(
            policy,
            update,
            total_steps,
            actor_optimizer,
            critic_optimizer,
            env_rng,
            value_normalizer,
            reference_policy,
        ),
    )


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
