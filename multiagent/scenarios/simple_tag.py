"""Course scenario: guide the prey to the shelter from rendered images."""

import numpy as np

from multiagent.core import Agent, Border, Check, Landmark, World
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    """One rule-based predator, one learned prey, and three fixed obstacles."""

    DEFAULT_OBSTACLES = np.array(
        [[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]], dtype=np.float32
    )

    def make_world(self):
        world = World()
        world.dim_c = 2

        world.agents = [Agent(), Agent()]
        for index, agent in enumerate(world.agents):
            agent.name = f"agent {index}"
            agent.collide = True
            agent.silent = True
            agent.adversary = index == 0
            agent.size = 0.025
            agent.accel = 0.85 if agent.adversary else 1.0
            agent.max_speed = 0.28 if agent.adversary else 0.25

        world.landmarks = [Landmark() for _ in range(3)]
        for index, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark {index}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False

        world.check = [Check()]
        shelter = world.check[0]
        shelter.name = "checkpoint 0"
        shelter.collide = False
        shelter.movable = False
        shelter.size = 0.05
        shelter.boundary = False
        shelter.shape = [
            [-0.025, -0.025],
            [0.025, -0.025],
            [0.025, 0.025],
            [-0.025, 0.025],
        ]

        world.borders = [Border() for _ in range(80)]
        for index, border in enumerate(world.borders):
            border.name = f"border {index}"
            border.collide = True
            border.movable = False
            border.size = 0.15
            border.boundary = True
            border.shape = [
                [-0.05, -0.05],
                [0.05, -0.05],
                [0.05, 0.05],
                [-0.05, 0.05],
            ]

        self.reset_world(world)
        return world

    def reset_world(self, world, agent_pos=None, check_pos=None, obstacles=None):
        predator, prey = world.agents
        predator.color = np.array([0.85, 0.35, 0.35])
        prey.color = np.array([0.35, 0.85, 0.35])
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
        for border in world.borders:
            border.color = np.array([0.8, 0.4, 0.4])
        world.check[0].color = np.array([0.8, 0.6, 0.8])

        if agent_pos is None:
            agent_pos = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        if agent_pos.shape != (2, 2):
            raise ValueError("agent_pos must contain [predator_position, prey_position]")

        for agent, position in zip(world.agents, agent_pos):
            agent.state.p_pos = position.copy()
            agent.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
            agent.action.u = np.zeros(world.dim_p, dtype=np.float32)
            agent.action.c = np.zeros(world.dim_c, dtype=np.float32)

        obstacle_positions = (
            self.DEFAULT_OBSTACLES
            if obstacles is None
            else np.asarray(obstacles, dtype=np.float32)
        )
        if obstacle_positions.shape != (3, 2):
            raise ValueError("obstacles must have shape (3, 2)")
        for landmark, position in zip(world.landmarks, obstacle_positions):
            landmark.state.p_pos = position.copy()
            landmark.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

        shelter_position = (
            np.array([-0.5, -0.5], dtype=np.float32)
            if check_pos is None
            else np.asarray(check_pos, dtype=np.float32)
        )
        world.check[0].state.p_pos = shelter_position.copy()
        world.check[0].state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

        border_positions = []
        border_positions.extend([[-0.95 + 0.1 * i, -1.0] for i in range(20)])
        border_positions.extend([[1.0, -0.95 + 0.1 * i] for i in range(20)])
        border_positions.extend([[0.95 - 0.1 * i, 1.0] for i in range(20)])
        border_positions.extend([[-1.0, 0.95 - 0.1 * i] for i in range(20)])
        for border, position in zip(world.borders, border_positions):
            border.state.p_pos = np.asarray(position, dtype=np.float32)
            border.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

        world.previous_goal_distance = self.goal_distance(world)
        world.previous_predator_distance = float(
            np.linalg.norm(prey.state.p_pos - predator.state.p_pos)
        )
        world.previous_obstacle_distance = float(
            min(
                np.linalg.norm(prey.state.p_pos - obstacle.state.p_pos)
                for obstacle in world.landmarks
            )
        )
        world.previous_prey_action = np.zeros(world.dim_p, dtype=np.float32)
        world.outcome = "running"

    @staticmethod
    def is_collision(entity_a, entity_b):
        distance = np.linalg.norm(entity_a.state.p_pos - entity_b.state.p_pos)
        return bool(distance < entity_a.size + entity_b.size)

    @staticmethod
    def good_agents(world):
        return [agent for agent in world.agents if not agent.adversary]

    @staticmethod
    def adversaries(world):
        return [agent for agent in world.agents if agent.adversary]

    def goal_distance(self, world):
        prey = self.good_agents(world)[0]
        return float(np.linalg.norm(prey.state.p_pos - world.check[0].state.p_pos))

    def outcome(self, world):
        prey = self.good_agents(world)[0]
        predator = self.adversaries(world)[0]

        if self.is_collision(prey, predator):
            return "predator_collision"
        if any(self.is_collision(prey, obstacle) for obstacle in world.landmarks):
            return "obstacle_collision"
        if self.is_collision(prey, world.check[0]):
            return "success"
        return "running"

    def reward(self, agent, world):
        if agent.adversary:
            return 0.0

        current_goal_distance = self.goal_distance(world)
        previous_goal_distance = getattr(
            world, "previous_goal_distance", current_goal_distance
        )
        # Potential-based progress supplies a useful signal before a terminal
        # success or collision is observed.
        reward = 10.0 * (previous_goal_distance - current_goal_distance) - 0.02

        predator = self.adversaries(world)[0]
        predator_distance = float(
            np.linalg.norm(agent.state.p_pos - predator.state.p_pos)
        )
        previous_predator_distance = getattr(
            world, "previous_predator_distance", predator_distance
        )
        predator_safety_distance = 0.35
        if predator_distance < predator_safety_distance:
            danger = (
                predator_safety_distance - predator_distance
            ) / predator_safety_distance
            reward -= 0.10 * danger * danger
            reward += 1.5 * (
                predator_distance - previous_predator_distance
            )

        obstacle_distance = float(min(
            np.linalg.norm(agent.state.p_pos - obstacle.state.p_pos)
            for obstacle in world.landmarks
        ))
        previous_obstacle_distance = getattr(
            world, "previous_obstacle_distance", obstacle_distance
        )
        obstacle_safety_distance = 0.14
        if obstacle_distance < obstacle_safety_distance:
            danger = (
                obstacle_safety_distance - obstacle_distance
            ) / obstacle_safety_distance
            reward -= 0.10 * danger * danger
            reward += 2.0 * min(
                0.0, obstacle_distance - previous_obstacle_distance
            )

        # Near the shelter, encourage a radial approach and remove tangential
        # velocity. This prevents a deterministic acceleration policy from
        # settling into a stable orbit just outside the success radius.
        docking_distance = 0.25
        if 1e-8 < current_goal_distance < docking_distance:
            goal_offset = world.check[0].state.p_pos - agent.state.p_pos
            goal_direction = goal_offset / current_goal_distance
            desired_speed = min(0.20, 1.5 * current_goal_distance)
            desired_velocity = desired_speed * goal_direction
            velocity_error = agent.state.p_vel - desired_velocity
            reward -= float(np.dot(velocity_error, velocity_error))

        # The policy outputs acceleration. A small change cost discourages
        # left-right oscillation without preventing a necessary evasive turn.
        previous_action = getattr(
            world, "previous_prey_action", np.zeros(world.dim_p)
        )
        action_change = agent.action.u - previous_action
        reward -= 0.005 * float(np.dot(action_change, action_change))

        result = self.outcome(world)
        if result == "success":
            reward += 50.0
        elif result in {"obstacle_collision", "predator_collision"}:
            reward -= 35.0
        return float(reward)

    def is_done(self, agent, world):
        del agent
        return self.outcome(world) != "running"

    def info(self, agent, world):
        del agent
        prey = self.good_agents(world)[0]
        predator = self.adversaries(world)[0]
        obstacle_distance = min(
            np.linalg.norm(prey.state.p_pos - obstacle.state.p_pos)
            for obstacle in world.landmarks
        )
        return {
            "outcome": self.outcome(world),
            "goal_distance": self.goal_distance(world),
            "predator_distance": float(
                np.linalg.norm(prey.state.p_pos - predator.state.p_pos)
            ),
            "obstacle_distance": float(obstacle_distance),
        }

    def benchmark_data(self, agent, world):
        return self.info(agent, world)

    def observation(self, agent, world):
        """Legacy simulator observation; the learned policy must not consume it."""

        other_agent = next(other for other in world.agents if other is not agent)
        relative_obstacles = [
            obstacle.state.p_pos - agent.state.p_pos
            for obstacle in world.landmarks
        ]
        return np.concatenate(
            [
                agent.state.p_pos,
                other_agent.state.p_pos - agent.state.p_pos,
                world.check[0].state.p_pos - agent.state.p_pos,
                *relative_obstacles,
                agent.state.p_vel,
                other_agent.state.p_vel,
            ]
        ).astype(np.float32)
