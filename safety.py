"""Lightweight predictive safety filter for the learned acceleration policy."""

from dataclasses import dataclass

import numpy as np

from vision import Detection


@dataclass
class SafetyResult:
    action: np.ndarray
    intervened: bool
    nominal_clearance: float
    selected_clearance: float


class PredictiveSafetyLayer:
    """Replace only actions predicted to cause an imminent collision.

    All positions and velocities supplied to this class come from RGB image
    detections. The short forward model is used for action screening only; the
    neural policy remains responsible for the primary navigation command.
    """

    def __init__(self, dt: float, horizon: int = 6):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.prey_accel = 1.0
        self.prey_max_speed = 0.25
        self.predator_accel = 0.85
        self.predator_max_speed = 0.28
        self.predator_safe_distance = 0.075
        self.obstacle_safe_distance = 0.09
        self.arena_limit = 0.90
        self.correction_count = 0
        self.step_count = 0

        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        self.base_candidates = np.concatenate(
            (
                np.zeros((1, 2), dtype=np.float32),
                0.5 * directions,
                directions,
            ),
            axis=0,
        ).astype(np.float32)

    def reset(self) -> None:
        self.correction_count = 0
        self.step_count = 0

    @staticmethod
    def _limit_speed(velocity: np.ndarray, maximum: float) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed > maximum:
            return velocity * (maximum / speed)
        return velocity

    def _minimum_clearance(
        self,
        action: np.ndarray,
        detection: Detection,
        prey_velocity: np.ndarray,
        predator_velocity: np.ndarray,
    ) -> float:
        prey_position = detection.prey.astype(np.float32).copy()
        predator_position = detection.predator.astype(np.float32).copy()
        prey_velocity = np.asarray(prey_velocity, dtype=np.float32).copy()
        predator_velocity = np.asarray(predator_velocity, dtype=np.float32).copy()
        minimum_clearance = float("inf")

        for _ in range(self.horizon):
            prey_velocity += action * self.prey_accel * self.dt
            prey_velocity = self._limit_speed(
                prey_velocity, self.prey_max_speed
            )

            pursuit = prey_position - predator_position
            pursuit_norm = float(np.linalg.norm(pursuit))
            if pursuit_norm > 1e-8:
                predator_velocity += (
                    pursuit / pursuit_norm
                ) * self.predator_accel * self.dt
            predator_velocity = self._limit_speed(
                predator_velocity, self.predator_max_speed
            )

            prey_position += prey_velocity * self.dt
            predator_position += predator_velocity * self.dt

            predator_clearance = (
                float(np.linalg.norm(prey_position - predator_position))
                - self.predator_safe_distance
            )
            obstacle_clearance = min(
                float(np.linalg.norm(prey_position - obstacle))
                - self.obstacle_safe_distance
                for obstacle in detection.obstacles
            )
            border_clearance = self.arena_limit - float(
                np.max(np.abs(prey_position))
            )
            minimum_clearance = min(
                minimum_clearance,
                predator_clearance,
                obstacle_clearance,
                border_clearance,
            )
        return minimum_clearance

    def filter(
        self,
        policy_action: np.ndarray,
        detection: Detection,
        prey_velocity: np.ndarray,
        predator_velocity: np.ndarray,
    ) -> SafetyResult:
        """Return the policy action unchanged unless it predicts danger."""

        policy_action = np.clip(
            np.asarray(policy_action, dtype=np.float32), -1.0, 1.0
        )
        self.step_count += 1
        nominal_clearance = self._minimum_clearance(
            policy_action, detection, prey_velocity, predator_velocity
        )
        if nominal_clearance >= 0.0:
            return SafetyResult(
                policy_action, False, nominal_clearance, nominal_clearance
            )

        braking = -np.asarray(prey_velocity, dtype=np.float32)
        braking_norm = float(np.linalg.norm(braking))
        if braking_norm > 1e-8:
            braking /= braking_norm
        candidates = np.concatenate(
            (
                policy_action.reshape(1, 2),
                braking.reshape(1, 2),
                self.base_candidates,
            ),
            axis=0,
        )

        scored = []
        for candidate in candidates:
            clearance = self._minimum_clearance(
                candidate, detection, prey_velocity, predator_velocity
            )
            distance = float(np.sum((candidate - policy_action) ** 2))
            scored.append((clearance, distance, candidate))

        safe = [entry for entry in scored if entry[0] >= 0.0]
        if safe:
            selected_clearance, _, selected_action = min(
                safe, key=lambda entry: entry[1]
            )
        else:
            selected_clearance, _, selected_action = max(
                scored, key=lambda entry: (entry[0], -entry[1])
            )

        intervened = not np.allclose(selected_action, policy_action)
        self.correction_count += int(intervened)
        return SafetyResult(
            np.asarray(selected_action, dtype=np.float32),
            intervened,
            nominal_clearance,
            selected_clearance,
        )
