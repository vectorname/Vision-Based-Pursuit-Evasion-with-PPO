"""Readable PPO implementation for the image-based pursuit-evasion task."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.PPO.model import ActorCritic


@dataclass
class RolloutBuffer:
    """Short-lived on-policy trajectories used for one PPO update."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    episode_ends: list[bool] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_probability: float,
        reward: float,
        value: float,
        next_value: float,
        terminated: bool,
        episode_end: bool,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probabilities.append(float(log_probability))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.episode_ends.append(bool(episode_end))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.values.clear()
        self.next_values.clear()
        self.terminated.clear()
        self.episode_ends.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPO:
    """Continuous-action PPO with GAE and minibatch updates."""

    def __init__(self, input_dim: int, action_dim: int, cfg):
        self.device = self._resolve_device(cfg.device)
        model_arguments = dict(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_size=cfg.hidden_size,
            initial_action_std=cfg.initial_action_std,
            min_log_std=cfg.min_log_std,
            max_log_std=cfg.max_log_std,
        )
        self.policy = ActorCritic(**model_arguments).to(self.device)
        self.old_policy = ActorCritic(**model_arguments).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.learning_rate
        )
        self.buffer = RolloutBuffer()

        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_ratio = cfg.clip_ratio
        self.update_epochs = cfg.update_epochs
        self.batch_size = cfg.batch_size
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.target_kl = cfg.target_kl

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_probability, value = self.old_policy.act(
                state_tensor, deterministic=deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_probability.item()),
            float(value.item()),
        )

    def value(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return float(self.old_policy.value(state_tensor).item())

    def set_learning_rate(self, learning_rate: float) -> None:
        """Change the optimizer rate when fine-tuning a checkpoint."""

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def set_action_std(self, action_std: float) -> None:
        """Reset exploration noise for a fine-tuning stage."""

        if action_std <= 0:
            raise ValueError("action_std must be positive")
        log_std = math.log(action_std)
        with torch.no_grad():
            self.policy.log_std.fill_(log_std)
            self.old_policy.log_std.fill_(log_std)

    def _advantages_and_returns(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        next_values = np.asarray(self.buffer.next_values, dtype=np.float32)
        terminated = np.asarray(self.buffer.terminated, dtype=np.float32)
        episode_ends = np.asarray(self.buffer.episode_ends, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            bootstrap = 1.0 - terminated[index]
            delta = (
                rewards[index]
                + self.gamma * next_values[index] * bootstrap
                - values[index]
            )
            continue_trajectory = 1.0 - episode_ends[index]
            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * continue_trajectory
                * gae
            )
            advantages[index] = gae

        returns = advantages + values
        advantage_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        return_tensor = torch.as_tensor(
            returns, dtype=torch.float32, device=self.device
        )
        advantage_tensor = (
            advantage_tensor - advantage_tensor.mean()
        ) / (advantage_tensor.std(unbiased=False) + 1e-8)
        return advantage_tensor, return_tensor

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {}

        states = torch.as_tensor(
            np.asarray(self.buffer.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.asarray(self.buffer.actions), dtype=torch.float32, device=self.device
        )
        old_log_probabilities = torch.as_tensor(
            self.buffer.log_probabilities, dtype=torch.float32, device=self.device
        )
        advantages, returns = self._advantages_and_returns()

        sample_count = len(self.buffer)
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        update_count = 0
        stop_early = False

        for _ in range(self.update_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                new_log_probabilities, entropy, predicted_values = (
                    self.policy.evaluate_actions(
                        states[batch_indices], actions[batch_indices]
                    )
                )

                log_ratio = (
                    new_log_probabilities
                    - old_log_probabilities[batch_indices]
                )
                probability_ratio = log_ratio.exp()
                unclipped = probability_ratio * advantages[batch_indices]
                clipped = torch.clamp(
                    probability_ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * advantages[batch_indices]
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = F.smooth_l1_loss(
                    predicted_values, returns[batch_indices]
                )
                entropy_mean = entropy.mean()
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
                metrics["actor_loss"] += float(actor_loss.item())
                metrics["critic_loss"] += float(critic_loss.item())
                metrics["entropy"] += float(entropy_mean.item())
                metrics["approx_kl"] += float(approximate_kl.item())
                update_count += 1

                if approximate_kl.item() > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()
        self.buffer.clear()

        if update_count:
            metrics = {key: value / update_count for key, value in metrics.items()}
        return metrics

    def save(self, path, **training_state) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": self.old_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            **training_state,
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    def load(self, path, load_optimizer: bool = False) -> dict:
        checkpoint = torch.load(Path(path), map_location=self.device)
        state_dict = checkpoint.get("policy", checkpoint)
        self.policy.load_state_dict(state_dict)
        self.old_policy.load_state_dict(state_dict)
        self.old_policy.eval()
        if load_optimizer and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
            return {}
        return {
            key: value
            for key, value in checkpoint.items()
            if key not in {"policy", "optimizer"}
        }
