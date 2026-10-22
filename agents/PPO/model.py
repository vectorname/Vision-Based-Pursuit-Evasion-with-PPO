"""Neural networks used by the continuous-action PPO agent."""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Separate actor and critic networks with a tanh-squashed Gaussian policy."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        initial_action_std: float = 0.6,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        initial_log_std = math.log(initial_action_std)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self.apply(self._orthogonal_init)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    @staticmethod
    def _orthogonal_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)

    def _distribution(self, states: torch.Tensor) -> Normal:
        means = self.actor(states)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        standard_deviation = log_std.exp().expand_as(means)
        return Normal(means, standard_deviation)

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.pow(2) + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def act(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(states)
        raw_actions = distribution.mean if deterministic else distribution.sample()
        actions = torch.tanh(raw_actions)
        log_probabilities = self._squashed_log_prob(
            distribution, raw_actions, actions
        )
        values = self.critic(states).squeeze(-1)
        return actions, log_probabilities, values

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(states)
        bounded_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(bounded_actions)
        log_probabilities = self._squashed_log_prob(
            distribution, raw_actions, bounded_actions
        )
        # The base-distribution entropy is a stable exploration proxy.
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probabilities, entropy, values

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)

    def forward(self, states: torch.Tensor):
        """Return deterministic bounded actions and values for inspection."""

        return torch.tanh(self.actor(states)), self.value(states)
