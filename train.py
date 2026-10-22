"""Train the image-state PPO controller on randomized course scenarios."""

import argparse
import csv
import datetime as dt
import json
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agents.PPO.agent import PPO
from config import PPOConfig
from env_utils import (
    configure_difficulty,
    curriculum_difficulty,
    make_env,
    policy_action_to_env_action,
    prey_outcome,
    render_training_rgb,
    reset_with_layout,
    sample_random_layout,
)
from evaluate import evaluate_policy
from vision import ImageDetector, StateBuilder


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(
    cfg: PPOConfig,
    run_directory: Path,
    resume: Optional[Path] = None,
    learning_rate_override: Optional[float] = None,
    action_std_override: Optional[float] = None,
) -> None:
    set_random_seed(cfg.seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, ensure_ascii=False)

    env = make_env()
    detector = ImageDetector()
    state_builder = StateBuilder(env.world.dt, cfg.velocity_smoothing)
    agent = PPO(cfg.state_dim, cfg.action_dim, cfg)
    training_state = {}
    if resume is not None:
        training_state = agent.load(resume, load_optimizer=True)
        if learning_rate_override is not None:
            agent.set_learning_rate(learning_rate_override)
        if action_std_override is not None:
            agent.set_action_std(action_std_override)

    total_steps = int(training_state.get("total_steps", 0))
    episode = int(training_state.get("episode", 0))
    # A resumed run should not replay the exact layout sequence from step zero.
    rng_seed = cfg.seed if total_steps == 0 else cfg.seed + total_steps
    rng = np.random.default_rng(rng_seed)
    next_evaluation = (
        total_steps // cfg.evaluation_interval + 1
    ) * cfg.evaluation_interval
    next_checkpoint = (
        total_steps // cfg.checkpoint_interval + 1
    ) * cfg.checkpoint_interval
    best_success_rate = float(training_state.get("best_success_rate", -1.0))
    train_log = run_directory / "train_log.csv"
    update_log = run_directory / "update_log.csv"
    evaluation_log = run_directory / "evaluation_log.csv"
    recent_rewards = deque(maxlen=20)
    recent_outcomes = deque(maxlen=20)

    if resume is not None:
        # Preserve a usable best model in the new run even if later evaluations
        # do not surpass the score stored in the resumed checkpoint.
        agent.save(
            run_directory / "best_model.pt",
            total_steps=total_steps,
            episode=episode,
            best_success_rate=best_success_rate,
        )

    while total_steps < cfg.total_training_steps:
        episode += 1
        difficulty = curriculum_difficulty(
            total_steps,
            cfg.curriculum_steps,
            cfg.curriculum_warmup_steps,
        )
        minimum_predator_distance = configure_difficulty(env, cfg, difficulty)
        layout = sample_random_layout(
            rng,
            minimum_predator_distance=minimum_predator_distance,
        )
        reset_with_layout(env, layout)

        detection = detector.detect(render_training_rgb(env))
        state = state_builder.reset(detection)
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

        for episode_step in range(1, cfg.max_episode_steps + 1):
            action, log_probability, value = agent.select_action(state)
            env_action = policy_action_to_env_action(action)
            _, rewards, _, info = env.step(env_action)

            next_detection = detector.detect(
                render_training_rgb(env), previous=detection
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
            next_state = state_builder.update(next_detection)
            environment_outcome = prey_outcome(info)
            terminated = environment_outcome != "running"
            truncated = episode_step == cfg.max_episode_steps and not terminated
            episode_end = terminated or truncated

            reward = float(rewards[1])
            if truncated:
                reward += cfg.timeout_penalty
            next_value = 0.0 if terminated else agent.value(next_state)
            agent.buffer.add(
                state=state,
                action=action,
                log_probability=log_probability,
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                episode_end=episode_end,
            )

            state = next_state
            detection = next_detection
            episode_reward += reward
            total_steps += 1

            if len(agent.buffer) >= cfg.rollout_steps:
                metrics = agent.update()
                append_csv(update_log, {"total_steps": total_steps, **metrics})

                if total_steps >= next_checkpoint:
                    agent.save(
                        run_directory / "latest_checkpoint.pt",
                        total_steps=total_steps,
                        episode=episode,
                        best_success_rate=best_success_rate,
                    )
                    next_checkpoint += cfg.checkpoint_interval

            if episode_end or total_steps >= cfg.total_training_steps:
                outcome = environment_outcome if terminated else "timeout"
                break

        append_csv(
            train_log,
            {
                "episode": episode,
                "total_steps": total_steps,
                "reward": episode_reward,
                "steps": episode_step,
                "outcome": outcome,
                "difficulty": difficulty,
                "goal_progress": initial_goal_distance - final_goal_distance,
                "path_length": path_length,
                "minimum_predator_distance": minimum_predator_distance,
                "detection_failures": detection_failures,
            },
        )
        recent_rewards.append(episode_reward)
        recent_outcomes.append(outcome)

        if episode % cfg.log_every_episodes == 0:
            recent_success_rate = recent_outcomes.count("success") / len(
                recent_outcomes
            )
            print(
                f"episode={episode} steps={total_steps}/{cfg.total_training_steps} "
                f"mean_reward_20={np.mean(recent_rewards):.2f} "
                f"success_20={recent_success_rate:.0%} outcome={outcome} "
                f"difficulty={difficulty:.2f}"
            )

        if total_steps >= next_evaluation:
            summary = evaluate_policy(
                agent,
                cfg,
                episodes=cfg.evaluation_episodes,
                seed=cfg.evaluation_seed,
                use_software_render=True,
            )
            append_csv(evaluation_log, {"total_steps": total_steps, **summary})
            print(
                f"evaluation steps={total_steps} "
                f"success_rate={summary['success_rate']:.2%}"
            )
            if summary["success_rate"] > best_success_rate:
                best_success_rate = summary["success_rate"]
                agent.save(
                    run_directory / "best_model.pt",
                    total_steps=total_steps,
                    episode=episode,
                    best_success_rate=best_success_rate,
                )
            next_evaluation += cfg.evaluation_interval

    if len(agent.buffer):
        metrics = agent.update()
        append_csv(update_log, {"total_steps": total_steps, **metrics})
    agent.save(
        run_directory / "last_model.pt",
        total_steps=total_steps,
        episode=episode,
        best_success_rate=best_success_rate,
    )
    if best_success_rate < 0.0:
        agent.save(
            run_directory / "best_model.pt",
            total_steps=total_steps,
            episode=episode,
            best_success_rate=best_success_rate,
        )
    env.close()
    print(f"Training complete. Results saved to {run_directory}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--action-std", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PPOConfig()
    if args.steps is not None:
        cfg.total_training_steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_root is not None:
        cfg.output_root = str(args.output_root)
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.action_std is not None:
        cfg.initial_action_std = args.action_std

    run_name = dt.datetime.now().strftime("run_%Y%m%d-%H%M%S")
    run_directory = Path(cfg.output_root) / run_name
    train(
        cfg,
        run_directory,
        resume=args.resume,
        learning_rate_override=args.learning_rate,
        action_std_override=args.action_std,
    )


if __name__ == "__main__":
    main()
