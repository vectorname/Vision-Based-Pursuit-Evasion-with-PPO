"""Run a trained policy in a visible course environment."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from agents.PPO.agent import PPO
from config import PPOConfig
from env_utils import (
    DEFAULT_OBSTACLES,
    Layout,
    make_env,
    policy_action_to_env_action,
    prey_outcome,
    render_rgb,
    reset_with_layout,
    sample_random_layout,
)
from vision import ImageDetector, StateBuilder, draw_detections
from safety import PredictiveSafetyLayer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoint" / "best_model.pt"


def requested_layout(args, rng: np.random.Generator) -> Layout:
    supplied = (args.predator, args.prey, args.goal)
    if all(value is None for value in supplied):
        return sample_random_layout(rng)
    if any(value is None for value in supplied):
        raise ValueError("--predator, --prey, and --goal must be supplied together")
    return Layout(
        predator=np.asarray(args.predator, dtype=np.float32),
        prey=np.asarray(args.prey, dtype=np.float32),
        goal=np.asarray(args.goal, dtype=np.float32),
        obstacles=DEFAULT_OBSTACLES.copy(),
    )


def run_demo(args) -> None:
    cfg = PPOConfig(seed=args.seed, max_episode_steps=args.max_steps)
    agent = PPO(cfg.state_dim, cfg.action_dim, cfg)
    agent.load(args.model)

    env = make_env()
    detector = ImageDetector()
    state_builder = StateBuilder(env.world.dt, cfg.velocity_smoothing)
    safety_layer = PredictiveSafetyLayer(env.world.dt, cfg.safety_horizon)
    rng = np.random.default_rng(args.seed)

    for episode in range(1, args.episodes + 1):
        layout = requested_layout(args, rng)
        reset_with_layout(env, layout)
        frame = render_rgb(env)
        detection = detector.detect(frame)
        state = state_builder.reset(detection)
        safety_layer.reset()
        outcome = "timeout"
        total_reward = 0.0

        for step in range(1, cfg.max_episode_steps + 1):
            policy_action, _, _ = agent.select_action(state, deterministic=True)
            if args.no_safety:
                action = policy_action
            else:
                safety_result = safety_layer.filter(
                    policy_action,
                    detection,
                    state_builder.prey_velocity,
                    state_builder.predator_velocity,
                )
                action = safety_result.action
            _, rewards, _, info = env.step(policy_action_to_env_action(action))
            frame = render_rgb(env)
            next_detection = detector.detect(frame, previous=detection)
            state = state_builder.update(next_detection)
            detection = next_detection
            total_reward += float(rewards[1])

            env.render("human")
            if args.show_detections:
                diagnostic = draw_detections(frame, detection)
                cv2.imshow(
                    "Image-derived detections",
                    cv2.cvtColor(diagnostic, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)

            environment_outcome = prey_outcome(info)
            if environment_outcome != "running":
                outcome = environment_outcome
                break
            if args.delay > 0:
                time.sleep(args.delay)

        print(
            f"episode={episode} outcome={outcome} "
            f"steps={step} reward={total_reward:.2f} "
            f"safety_corrections={safety_layer.correction_count}"
        )

    cv2.destroyAllWindows()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--show-detections", action="store_true")
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--predator", nargs=2, type=float)
    parser.add_argument("--prey", nargs=2, type=float)
    parser.add_argument("--goal", nargs=2, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
