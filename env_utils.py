"""Environment construction and task-specific helper functions."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


DEFAULT_OBSTACLES = np.array(
    [[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]], dtype=np.float32
)


@dataclass
class Layout:
    predator: np.ndarray
    prey: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray


def make_env() -> MultiAgentEnv:
    """Create the course environment with outcome information enabled."""

    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    return MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=scenario.info,
        done_callback=scenario.is_done,
        shared_viewer=True,
    )


def _far_from_points(position: np.ndarray, points: np.ndarray, margin: float) -> bool:
    return bool(np.all(np.linalg.norm(points - position, axis=1) > margin))


def sample_random_layout(
    rng: np.random.Generator,
    obstacles: Optional[np.ndarray] = None,
    minimum_predator_distance: float = 0.30,
) -> Layout:
    """Sample non-overlapping initial positions inside the visible arena."""

    obstacles = DEFAULT_OBSTACLES if obstacles is None else np.asarray(obstacles, dtype=np.float32)
    for _ in range(10_000):
        predator = rng.uniform(-0.75, 0.75, size=2).astype(np.float32)
        prey = rng.uniform(-0.75, 0.75, size=2).astype(np.float32)
        goal = rng.uniform(-0.75, 0.75, size=2).astype(np.float32)

        if not _far_from_points(predator, obstacles, 0.12):
            continue
        if not _far_from_points(prey, obstacles, 0.12):
            continue
        if not _far_from_points(goal, obstacles, 0.14):
            continue
        if np.linalg.norm(prey - predator) <= minimum_predator_distance:
            continue
        if np.linalg.norm(prey - goal) <= 0.40:
            continue
        if np.linalg.norm(predator - goal) <= 0.15:
            continue

        return Layout(predator, prey, goal, obstacles.copy())
    raise RuntimeError("Could not sample a valid layout after 10,000 attempts")


def curriculum_difficulty(
    total_steps: int,
    curriculum_steps: int,
    warmup_steps: int = 0,
) -> float:
    """Return a smooth difficulty value from 0 (easy) to 1 (normal)."""

    if curriculum_steps <= 0:
        return 1.0
    curriculum_progress = (total_steps - warmup_steps) / curriculum_steps
    return float(np.clip(curriculum_progress, 0.0, 1.0))


def configure_difficulty(env: MultiAgentEnv, cfg, difficulty: float) -> float:
    """Apply curriculum settings and return the required spawn distance."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    predator = env.world.agents[0]
    predator.max_speed = float(
        np.interp(
            difficulty,
            [0.0, 1.0],
            [cfg.initial_predator_speed, cfg.final_predator_speed],
        )
    )
    predator.accel = float(
        np.interp(
            difficulty,
            [0.0, 1.0],
            [cfg.initial_predator_accel, cfg.final_predator_accel],
        )
    )
    return float(
        np.interp(
            difficulty,
            [0.0, 1.0],
            [cfg.initial_predator_distance, cfg.final_predator_distance],
        )
    )


def reset_with_layout(env: MultiAgentEnv, layout: Layout):
    agent_positions = np.stack((layout.predator, layout.prey))
    return env.reset(agent_positions, layout.goal, layout.obstacles)


def policy_action_to_env_action(action: np.ndarray) -> np.ndarray:
    """Convert [ax, ay] in [-1, 1] to [noop, right, left, up, down]."""

    ax, ay = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.array(
        [0.0, max(ax, 0.0), max(-ax, 0.0), max(ay, 0.0), max(-ay, 0.0)],
        dtype=np.float32,
    )


def render_rgb(env: MultiAgentEnv) -> np.ndarray:
    """Return the RGB image produced by the course environment renderer."""

    return np.asarray(env.render("rgb_array")[0])


def render_training_rgb(env: MultiAgentEnv) -> np.ndarray:
    """Render a matching 800x800 observation without a GUI window.

    This fast renderer is restricted to training and its internal validation.
    Demo and formal command-line evaluation use :func:`render_rgb`, which calls
    the original environment renderer.
    """

    import cv2

    size = 800
    pixels_per_world_unit = size / 2.0
    background = getattr(env, "_software_render_background", None)
    if background is None:
        background = np.full((size, size, 3), 255, dtype=np.uint8)

    def center(position: np.ndarray) -> tuple[int, int]:
        x, y = np.asarray(position, dtype=np.float32)
        return (
            int(round((float(x) + 1.0) * pixels_per_world_unit)),
            int(round((1.0 - float(y)) * pixels_per_world_unit)),
        )

    if getattr(env, "_software_render_background", None) is None:
        # Arena walls never move, so draw them only once per environment.
        for border in env.world.borders:
            column, row = center(border.state.p_pos)
            half_width = int(round(0.05 * pixels_per_world_unit))
            cv2.rectangle(
                background,
                (column - half_width, row - half_width),
                (column + half_width, row + half_width),
                (204, 102, 102),
                thickness=-1,
            )
        env._software_render_background = background
    image = background.copy()

    for obstacle in env.world.landmarks:
        cv2.circle(
            image,
            center(obstacle.state.p_pos),
            int(round(obstacle.size * pixels_per_world_unit)),
            (64, 64, 64),
            thickness=-1,
        )

    shelter = env.world.check[0]
    column, row = center(shelter.state.p_pos)
    half_width = int(round(0.025 * pixels_per_world_unit))
    cv2.rectangle(
        image,
        (column - half_width, row - half_width),
        (column + half_width, row + half_width),
        (204, 153, 204),
        thickness=-1,
    )

    predator, prey = env.world.agents
    agent_radius = int(round(prey.size * pixels_per_world_unit))
    cv2.circle(
        image,
        center(predator.state.p_pos),
        agent_radius,
        (236, 172, 172),
        thickness=-1,
    )
    cv2.circle(
        image,
        center(prey.state.p_pos),
        agent_radius,
        (172, 236, 172),
        thickness=-1,
    )
    return image


def prey_outcome(info: dict) -> str:
    """Read the prey's explicit outcome from the legacy multi-agent info dict."""

    entries = info.get("n", [])
    if len(entries) < 2:
        return "running"
    return str(entries[1].get("outcome", "running"))
