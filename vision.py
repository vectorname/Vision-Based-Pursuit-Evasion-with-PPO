"""Extract the policy state exclusively from rendered RGB images."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class DetectionError(RuntimeError):
    """Raised when the first frame does not contain every required object."""


@dataclass
class Detection:
    """Object positions recovered from one rendered RGB frame."""

    prey: np.ndarray
    predator: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    valid: bool = True
    missing: Tuple[str, ...] = ()


class ImageDetector:
    """Detect the color-coded entities used by the course environment."""

    # Rendering blends the two agents with the white background at alpha=0.5.
    PREY_RGB = np.array([172, 236, 172], dtype=np.int16)
    PREDATOR_RGB = np.array([236, 172, 172], dtype=np.int16)
    GOAL_RGB = np.array([204, 153, 204], dtype=np.int16)
    OBSTACLE_RGB = np.array([64, 64, 64], dtype=np.int16)

    def __init__(
        self,
        color_tolerance: int = 18,
        min_area: int = 20,
        max_area_fraction: float = 0.02,
    ):
        self.color_tolerance = int(color_tolerance)
        self.min_area = int(min_area)
        self.max_area_fraction = float(max_area_fraction)

    @staticmethod
    def _pixel_to_world(center: Sequence[float], width: int, height: int) -> np.ndarray:
        """Convert an image center (column, row) to the world's [-1, 1] frame."""

        column, row = center
        x = 2.0 * float(column) / float(width) - 1.0
        y = 1.0 - 2.0 * float(row) / float(height)
        return np.array([x, y], dtype=np.float32)

    def _color_mask(self, image: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Create a color mask with OpenCV's optimized range operation."""

        lower = np.clip(rgb - self.color_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(rgb + self.color_tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(image, lower, upper)

    def _single_center(
        self, image: np.ndarray, rgb: np.ndarray
    ) -> list[np.ndarray]:
        """Find one uniquely colored entity without connected components."""

        mask = self._color_mask(image, rgb)
        moments = cv2.moments(mask, binaryImage=True)
        area = moments["m00"]
        max_area = image.shape[0] * image.shape[1] * self.max_area_fraction
        if not self.min_area <= area <= max_area:
            return []
        center = (moments["m10"] / area, moments["m01"] / area)
        height, width = image.shape[:2]
        return [self._pixel_to_world(center, width, height)]

    def _component_centers(self, image: np.ndarray, rgb: np.ndarray) -> list[np.ndarray]:
        mask = self._color_mask(image, rgb)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        components = []
        max_area = image.shape[0] * image.shape[1] * self.max_area_fraction
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if self.min_area <= area <= max_area:
                components.append((area, centroids[index]))

        components.sort(key=lambda item: item[0], reverse=True)
        height, width = image.shape[:2]
        return [self._pixel_to_world(center, width, height) for _, center in components]

    @staticmethod
    def _fallback(
        name: str,
        positions: list[np.ndarray],
        previous: Optional[Detection],
    ) -> tuple[np.ndarray, bool]:
        if positions:
            return positions[0], True
        if previous is not None:
            return np.array(getattr(previous, name), dtype=np.float32), False
        raise DetectionError(f"Could not detect {name} in the first frame")

    def detect(self, image: np.ndarray, previous: Optional[Detection] = None) -> Detection:
        """Detect all entities, falling back to the previous frame when possible."""

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB array with shape (height, width, 3)")

        prey, prey_ok = self._fallback(
            "prey", self._single_center(image, self.PREY_RGB), previous
        )
        predator, predator_ok = self._fallback(
            "predator", self._single_center(image, self.PREDATOR_RGB), previous
        )
        goal, goal_ok = self._fallback(
            "goal", self._single_center(image, self.GOAL_RGB), previous
        )

        obstacle_centers = self._component_centers(image, self.OBSTACLE_RGB)
        obstacles_ok = len(obstacle_centers) >= 3
        if obstacles_ok:
            obstacles = np.stack(obstacle_centers[:3]).astype(np.float32)
            # Fixed ordering prevents state entries from swapping between frames.
            order = np.lexsort((obstacles[:, 1], obstacles[:, 0]))
            obstacles = obstacles[order]
        elif previous is not None:
            obstacles = previous.obstacles.copy()
        else:
            raise DetectionError("Could not detect all three obstacles in the first frame")

        missing = []
        for name, detected in (
            ("prey", prey_ok),
            ("predator", predator_ok),
            ("goal", goal_ok),
            ("obstacles", obstacles_ok),
        ):
            if not detected:
                missing.append(name)

        return Detection(
            prey=prey,
            predator=predator,
            goal=goal,
            obstacles=obstacles,
            valid=not missing,
            missing=tuple(missing),
        )


class StateBuilder:
    """Build the 16-D policy state and estimate velocity from adjacent frames."""

    def __init__(self, dt: float, velocity_smoothing: float = 0.5):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not 0.0 <= velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        self.dt = float(dt)
        self.alpha = float(velocity_smoothing)
        self.previous: Optional[Detection] = None
        self.prey_velocity = np.zeros(2, dtype=np.float32)
        self.predator_velocity = np.zeros(2, dtype=np.float32)

    def reset(self, detection: Detection) -> np.ndarray:
        self.previous = detection
        self.prey_velocity.fill(0.0)
        self.predator_velocity.fill(0.0)
        return self._compose(detection)

    def update(self, detection: Detection) -> np.ndarray:
        if self.previous is None:
            return self.reset(detection)

        raw_prey_velocity = (detection.prey - self.previous.prey) / self.dt
        raw_predator_velocity = (detection.predator - self.previous.predator) / self.dt
        self.prey_velocity = (
            self.alpha * raw_prey_velocity
            + (1.0 - self.alpha) * self.prey_velocity
        )
        self.predator_velocity = (
            self.alpha * raw_predator_velocity
            + (1.0 - self.alpha) * self.predator_velocity
        )
        self.prey_velocity = np.clip(self.prey_velocity, -1.0, 1.0)
        self.predator_velocity = np.clip(self.predator_velocity, -1.0, 1.0)
        self.previous = detection
        return self._compose(detection)

    def _compose(self, detection: Detection) -> np.ndarray:
        obstacle_offsets = (detection.obstacles - detection.prey).reshape(-1)
        state = np.concatenate(
            (
                detection.goal - detection.prey,
                detection.prey,
                detection.predator - detection.prey,
                obstacle_offsets,
                self.prey_velocity,
                self.predator_velocity,
            )
        ).astype(np.float32)
        if state.shape != (16,):
            raise RuntimeError(f"Expected a 16-D state, got {state.shape}")
        return state


def draw_detections(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Return a copy of the frame with simple diagnostic markers."""

    output = image.copy()
    height, width = output.shape[:2]

    def world_to_pixel(position: np.ndarray) -> tuple[int, int]:
        column = int(round((float(position[0]) + 1.0) * 0.5 * width))
        row = int(round((1.0 - float(position[1])) * 0.5 * height))
        return column, row

    for name, position, color in (
        ("prey", detection.prey, (0, 180, 0)),
        ("predator", detection.predator, (220, 0, 0)),
        ("goal", detection.goal, (180, 0, 180)),
    ):
        center = world_to_pixel(position)
        cv2.circle(output, center, 10, color, 2)
        cv2.putText(
            output,
            name,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    for index, position in enumerate(detection.obstacles, start=1):
        center = world_to_pixel(position)
        cv2.circle(output, center, 10, (0, 0, 0), 2)
        cv2.putText(
            output,
            f"o{index}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return output
