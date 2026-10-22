"""Teach an existing PPO policy to enter the shelter without circling it.

The normal PPO objective is good at reaching the shelter neighbourhood, but
the environment has almost no velocity damping.  A deterministic policy can
therefore learn a stable orbit just outside the success radius.  This script
uses a small, image-state expert data set to teach the actor a velocity-aware
arrival manoeuvre while distilling its original actions away from the goal.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.PPO.agent import PPO
from config import PPOConfig
from env_utils import (
    configure_difficulty,
    make_env,
    policy_action_to_env_action,
    prey_outcome,
    render_training_rgb,
    reset_with_layout,
    sample_random_layout,
)
from safety import PredictiveSafetyLayer
from vision import ImageDetector, StateBuilder


def docking_expert(state: np.ndarray, dt: float) -> np.ndarray:
    """Return a bounded acceleration that approaches and brakes at the goal."""

    goal_offset = state[:2]
    distance = float(np.linalg.norm(goal_offset))
    if distance <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    goal_direction = goal_offset / distance
    desired_speed = min(0.20, 1.5 * distance)
    desired_velocity = desired_speed * goal_direction
    prey_velocity = state[12:14]
    return np.clip(
        (desired_velocity - prey_velocity) / dt, -1.0, 1.0
    ).astype(np.float32)


def collect_policy_states(
    agent: PPO,
    cfg: PPOConfig,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collect realistic image-derived states and the original policy actions."""

    env = make_env()
    configure_difficulty(env, cfg, difficulty=1.0)
    detector = ImageDetector()
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(episodes):
        reset_with_layout(env, sample_random_layout(rng))
        detection = detector.detect(render_training_rgb(env))
        state_builder = StateBuilder(env.world.dt, cfg.velocity_smoothing)
        state = state_builder.reset(detection)
        safety = PredictiveSafetyLayer(env.world.dt, cfg.safety_horizon)

        for _ in range(cfg.max_episode_steps):
            action, _, _ = agent.select_action(state, deterministic=True)
            states.append(state.copy())
            actions.append(action.copy())
            safe_action = safety.filter(
                action,
                detection,
                state_builder.prey_velocity,
                state_builder.predator_velocity,
            ).action
            _, _, _, info = env.step(
                policy_action_to_env_action(safe_action)
            )
            next_detection = detector.detect(
                render_training_rgb(env), previous=detection
            )
            state = state_builder.update(next_detection)
            detection = next_detection
            if prey_outcome(info) != "running":
                break

    simulation_dt = float(env.world.dt)
    env.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        simulation_dt,
    )


def make_docking_examples(
    source_states: np.ndarray,
    count: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create realistic near-goal states with varied velocities and directions."""

    indices = rng.integers(0, len(source_states), size=count)
    states = source_states[indices].copy()
    angles = rng.uniform(-np.pi, np.pi, size=count)
    distances = rng.uniform(0.075, 0.30, size=count)
    states[:, 0] = distances * np.cos(angles)
    states[:, 1] = distances * np.sin(angles)

    velocity_angles = rng.uniform(-np.pi, np.pi, size=count)
    speeds = 0.25 * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    states[:, 12] = speeds * np.cos(velocity_angles)
    states[:, 13] = speeds * np.sin(velocity_angles)
    actions = np.stack([docking_expert(state, dt) for state in states])
    return states.astype(np.float32), actions.astype(np.float32)


def fine_tune(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = PPOConfig(seed=args.seed)
    agent = PPO(cfg.state_dim, cfg.action_dim, cfg)
    training_state = agent.load(args.model, load_optimizer=False)
    states, original_actions, simulation_dt = collect_policy_states(
        agent, cfg, args.collect_episodes, args.seed
    )
    distances = np.linalg.norm(states[:, :2], axis=1)
    general_mask = distances >= args.docking_distance
    general_states = states[general_mask]
    general_actions = original_actions[general_mask]
    if len(general_states) == 0:
        raise RuntimeError("No states outside the docking region were collected")

    rng = np.random.default_rng(args.seed + 1)
    docking_states, docking_actions = make_docking_examples(
        states, args.docking_examples, dt=simulation_dt, rng=rng
    )
    device = agent.device
    actor_parameters = list(agent.policy.actor.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=args.learning_rate)

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    docking_states_t = tensor(docking_states)
    docking_actions_t = tensor(docking_actions)
    general_states_t = tensor(general_states)
    general_actions_t = tensor(general_actions)

    for update in range(1, args.updates + 1):
        docking_indices = torch.randint(
            len(docking_states_t), (args.batch_size,), device=device
        )
        general_indices = torch.randint(
            len(general_states_t), (args.batch_size,), device=device
        )
        predicted_docking = torch.tanh(
            agent.policy.actor(docking_states_t[docking_indices])
        )
        predicted_general = torch.tanh(
            agent.policy.actor(general_states_t[general_indices])
        )
        docking_loss = F.mse_loss(
            predicted_docking, docking_actions_t[docking_indices]
        )
        preservation_loss = F.mse_loss(
            predicted_general, general_actions_t[general_indices]
        )
        loss = docking_loss + args.preservation_weight * preservation_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
        optimizer.step()

        if update == 1 or update % 250 == 0 or update == args.updates:
            print(
                f"update={update}/{args.updates} "
                f"docking_loss={docking_loss.item():.5f} "
                f"preservation_loss={preservation_loss.item():.5f}"
            )

    agent.old_policy.load_state_dict(agent.policy.state_dict())
    agent.old_policy.eval()
    output_state = {
        **training_state,
        "docking_bc_updates": args.updates,
        "docking_bc_examples": args.docking_examples,
        "docking_bc_source": str(args.model),
    }
    agent.save(args.output, **output_state)
    print(
        f"Saved docking-aware checkpoint to {args.output} "
        f"({len(general_states)} preserved states, "
        f"{len(docking_states)} expert states)"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-episodes", type=int, default=120)
    parser.add_argument("--docking-examples", type=int, default=20_000)
    parser.add_argument("--docking-distance", type=float, default=0.30)
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--preservation-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=31_415)
    return parser.parse_args()


if __name__ == "__main__":
    fine_tune(parse_args())
