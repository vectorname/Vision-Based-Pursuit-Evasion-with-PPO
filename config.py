"""Configuration shared by training, evaluation, and demonstration scripts."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PPOConfig:
    """Hyperparameters for the image-based PPO controller."""

    seed: int = 15
    state_dim: int = 16
    action_dim: int = 2

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 10
    batch_size: int = 128
    # A shorter rollout produces more policy updates during small test runs.
    rollout_steps: int = 1024
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    total_training_steps: int = 500_000
    max_episode_steps: int = 300
    # A timeout must not be preferable to making a serious attempt at the goal.
    timeout_penalty: float = -25.0
    log_every_episodes: int = 10
    evaluation_interval: int = 25_000
    evaluation_episodes: int = 50
    evaluation_seed: int = 10_000
    checkpoint_interval: int = 5_000

    hidden_size: int = 128
    initial_action_std: float = 0.45
    min_log_std: float = -5.0
    max_log_std: float = 1.0

    velocity_smoothing: float = 0.5
    safety_horizon: int = 5

    # Curriculum: the final values are exactly the normal task difficulty.
    curriculum_warmup_steps: int = 10_000
    curriculum_steps: int = 80_000
    initial_predator_speed: float = 0.0
    final_predator_speed: float = 0.28
    initial_predator_accel: float = 0.0
    final_predator_accel: float = 0.85
    initial_predator_distance: float = 0.65
    final_predator_distance: float = 0.30

    output_root: str = "outputs"
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["output_root"] = str(Path(self.output_root))
        return result
