"""Evaluate a trained policy on reproducible randomized layouts."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from agents.PPO.agent import PPO
from config import PPOConfig
from env_utils import (
    configure_difficulty,
    make_env,
    policy_action_to_env_action,
    prey_outcome,
    render_rgb,
    render_training_rgb,
    reset_with_layout,
    sample_random_layout,
)
from vision import ImageDetector, StateBuilder
from safety import PredictiveSafetyLayer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoint" / "best_model.pt"


def evaluate_policy(
    agent: PPO,
    cfg: PPOConfig,
    episodes: int,
    seed: int,
    csv_path: Optional[Path] = None,
    use_safety_layer: bool = True,
    use_software_render: bool = False,
) -> dict:
    """Run deterministic episodes and return outcome statistics."""

    env = make_env()
    # Evaluation always measures the original, full-difficulty task.
    configure_difficulty(env, cfg, difficulty=1.0)
    detector = ImageDetector()
    state_builder = StateBuilder(env.world.dt, cfg.velocity_smoothing)
    safety_layer = PredictiveSafetyLayer(env.world.dt, cfg.safety_horizon)
    rng = np.random.default_rng(seed)
    rows = []
    frame_renderer = render_training_rgb if use_software_render else render_rgb

    for episode in range(episodes):
        layout = sample_random_layout(rng)
        reset_with_layout(env, layout)
        detection = detector.detect(frame_renderer(env))
        state = state_builder.reset(detection)
        safety_layer.reset()
        initial_goal_distance = float(
            np.linalg.norm(detection.goal - detection.prey)
        )
        final_goal_distance = initial_goal_distance
        path_length = 0.0
        minimum_predator_distance = float(
            np.linalg.norm(detection.predator - detection.prey)
        )
        episode_reward = 0.0
        detection_failures = 0
        outcome = "timeout"

        for step in range(1, cfg.max_episode_steps + 1):
            policy_action, _, _ = agent.select_action(state, deterministic=True)
            if use_safety_layer:
                safety_result = safety_layer.filter(
                    policy_action,
                    detection,
                    state_builder.prey_velocity,
                    state_builder.predator_velocity,
                )
                action = safety_result.action
            else:
                action = policy_action
            env_action = policy_action_to_env_action(action)
            _, rewards, _, info = env.step(env_action)

            next_detection = detector.detect(
                frame_renderer(env), previous=detection
            )
            detection_failures += int(not next_detection.valid)
            path_length += float(
                np.linalg.norm(next_detection.prey - detection.prey)
            )
            final_goal_distance = float(
                np.linalg.norm(next_detection.goal - next_detection.prey)
            )
            minimum_predator_distance = min(
                minimum_predator_distance,
                float(
                    np.linalg.norm(
                        next_detection.predator - next_detection.prey
                    )
                ),
            )
            state = state_builder.update(next_detection)
            detection = next_detection
            episode_reward += float(rewards[1])

            environment_outcome = prey_outcome(info)
            if environment_outcome != "running":
                outcome = environment_outcome
                break

        rows.append(
            {
                "episode": episode + 1,
                "reward": episode_reward,
                "steps": step,
                "outcome": outcome,
                "goal_progress": initial_goal_distance - final_goal_distance,
                "path_length": path_length,
                "path_efficiency": (
                    initial_goal_distance / max(path_length, 1e-6)
                    if outcome == "success"
                    else float("nan")
                ),
                "minimum_predator_distance": minimum_predator_distance,
                "safety_corrections": safety_layer.correction_count,
                "detection_failures": detection_failures,
                "predator_x": float(layout.predator[0]),
                "predator_y": float(layout.predator[1]),
                "prey_x": float(layout.prey[0]),
                "prey_y": float(layout.prey[1]),
                "goal_x": float(layout.goal[0]),
                "goal_y": float(layout.goal[1]),
            }
        )

    env.close()
    counts = Counter(row["outcome"] for row in rows)
    successful_steps = [row["steps"] for row in rows if row["outcome"] == "success"]
    successful_efficiencies = [
        row["path_efficiency"]
        for row in rows
        if row["outcome"] == "success"
    ]
    summary = {
        "episodes": episodes,
        "success": counts["success"],
        "obstacle_collision": counts["obstacle_collision"],
        "predator_collision": counts["predator_collision"],
        "timeout": counts["timeout"],
        "success_rate": counts["success"] / episodes if episodes else 0.0,
        "mean_success_steps": (
            float(np.mean(successful_steps)) if successful_steps else float("nan")
        ),
        "mean_path_efficiency": (
            float(np.mean(successful_efficiencies))
            if successful_efficiencies
            else float("nan")
        ),
        "mean_goal_progress": float(
            np.mean([row["goal_progress"] for row in rows])
        ),
        "safety_correction_rate": (
            sum(row["safety_corrections"] for row in rows)
            / max(sum(row["steps"] for row in rows), 1)
        ),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
    }

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return summary


def print_summary(summary: dict) -> None:
    print(f"Episodes: {summary['episodes']}")
    print(f"Success: {summary['success']}")
    print(f"Obstacle collisions: {summary['obstacle_collision']}")
    print(f"Predator collisions: {summary['predator_collision']}")
    print(f"Timeouts: {summary['timeout']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Mean successful steps: {summary['mean_success_steps']:.1f}")
    print(f"Mean path efficiency: {summary['mean_path_efficiency']:.3f}")
    print(f"Mean goal progress: {summary['mean_goal_progress']:.3f}")
    print(f"Safety correction rate: {summary['safety_correction_rate']:.2%}")
    print(f"Mean reward: {summary['mean_reward']:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"))
    parser.add_argument("--safety-horizon", type=int, default=None)
    parser.add_argument("--no-safety", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PPOConfig()
    if args.safety_horizon is not None:
        cfg.safety_horizon = args.safety_horizon
    agent = PPO(cfg.state_dim, cfg.action_dim, cfg)
    agent.load(args.model)
    summary = evaluate_policy(
        agent,
        cfg,
        episodes=args.episodes,
        seed=args.seed,
        csv_path=args.output,
        use_safety_layer=not args.no_safety,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
