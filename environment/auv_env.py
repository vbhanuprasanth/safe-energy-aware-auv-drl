import os
import yaml
import numpy as np
import gymnasium as gym

from gymnasium import spaces

from environment.dynamics import AUVDynamics
from environment.obstacles import Obstacle
from environment.rewards import RewardCalculator
from environment.energy import EnergyModel


class AUVEnvironment(gym.Env):

    """
    3D Autonomous Underwater Vehicle navigation environment.

    Features:
    - 3D navigation
    - Random starting positions
    - Goal reaching
    - Obstacle detection
    - Collision detection
    - Safety zones
    - Energy consumption
    - Boundary handling
    """

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # =================================================
        # LOAD CONFIGURATION
        # =================================================

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "config.yaml"
        )

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # =================================================
        # ENVIRONMENT SETTINGS
        # =================================================

        env_config = config["environment"]

        self.environment_size = np.array(
            [
                env_config["size"]["x"],
                env_config["size"]["y"],
                env_config["size"]["z"]
            ],
            dtype=np.float32
        )

        self.max_steps = int(
            env_config["max_steps"]
        )

        self.current_step = 0

        # =================================================
        # AUV SETTINGS
        # =================================================

        auv_config = config["auv"]

        self.start_position = np.array(
            auv_config["start_position"],
            dtype=np.float32
        )

        self.auv_radius = float(
            auv_config["radius"]
        )

        self.dynamics = AUVDynamics(
            speed=float(
                auv_config["speed"]
            )
        )

        # =================================================
        # GOAL SETTINGS
        # =================================================

        goal_config = config["goal"]

        self.goal_position = np.array(
            goal_config["position"],
            dtype=np.float32
        )

        self.goal_radius = float(
            goal_config["radius"]
        )

        # =================================================
        # SAFETY SETTINGS
        # =================================================

        safety_config = config["safety"]

        self.safety_distance = float(
            safety_config["safety_distance"]
        )

        self.max_nearby_obstacles = int(
            safety_config["max_nearby_obstacles"]
        )

        # =================================================
        # REWARD SYSTEM
        # =================================================

        self.reward_calculator = RewardCalculator()

        # =================================================
        # ENERGY SYSTEM
        # =================================================

        self.energy_model = EnergyModel(
            initial_energy=100.0
        )

        # =================================================
        # OBSTACLES
        # =================================================

        obstacle_config = config["obstacles"]

        self.obstacles = [

            Obstacle(
                [6.0, 6.0, 4.0],
                radius=float(
                    obstacle_config["radius"]
                )
            ),

            Obstacle(
                [10.0, 8.0, 5.0],
                radius=1.2
            ),

            Obstacle(
                [14.0, 12.0, 6.0],
                radius=1.0
            ),

            Obstacle(
                [8.0, 15.0, 3.0],
                radius=1.5
            ),

            Obstacle(
                [15.0, 16.0, 7.0],
                radius=1.0
            )
        ]

        # =================================================
        # ACTION SPACE
        # =================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # =================================================
        # OBSERVATION SPACE
        # =================================================

        obstacle_observation_size = (
            self.max_nearby_obstacles * 4
        )

        total_observation_size = (
            3
            + 3
            + 1
            + 1
            + obstacle_observation_size
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_observation_size,),
            dtype=np.float32
        )

        # =================================================
        # CURRENT AUV STATE
        # =================================================

        self.auv_position = (
            self.start_position.copy()
        )

    # =====================================================
    # GET OBSERVATION
    # =====================================================

    def _get_observation(self):

        normalized_auv_position = (
            self.auv_position
            / self.environment_size
        )

        relative_goal_position = (
            self.goal_position
            - self.auv_position
        )

        normalized_relative_goal = (
            relative_goal_position
            / self.environment_size
        )

        distance_to_goal = np.linalg.norm(
            relative_goal_position
        )

        maximum_possible_distance = np.linalg.norm(
            self.environment_size
        )

        normalized_distance_to_goal = (
            distance_to_goal
            / maximum_possible_distance
        )

        normalized_energy = (
            self.energy_model.remaining_energy
            / self.energy_model.initial_energy
        )

        nearby_obstacles = (
            self._get_nearby_obstacles()
        )

        normalized_obstacles = []

        for i in range(
            self.max_nearby_obstacles
        ):

            start_index = i * 4

            relative_x = (
                nearby_obstacles[start_index]
                / self.environment_size[0]
            )

            relative_y = (
                nearby_obstacles[start_index + 1]
                / self.environment_size[1]
            )

            relative_z = (
                nearby_obstacles[start_index + 2]
                / self.environment_size[2]
            )

            distance = (
                nearby_obstacles[start_index + 3]
                / maximum_possible_distance
            )

            normalized_obstacles.extend(
                [
                    relative_x,
                    relative_y,
                    relative_z,
                    distance
                ]
            )

        normalized_obstacles = np.array(
            normalized_obstacles,
            dtype=np.float32
        )

        observation = np.concatenate(
            [
                normalized_auv_position,
                normalized_relative_goal,

                np.array(
                    [
                        normalized_distance_to_goal
                    ],
                    dtype=np.float32
                ),

                np.array(
                    [
                        normalized_energy
                    ],
                    dtype=np.float32
                ),

                normalized_obstacles
            ]
        ).astype(np.float32)

        return observation

    # =====================================================
    # BOUNDARY CHECK
    # =====================================================

    def _is_out_of_bounds(self, position):

        return bool(
            np.any(position < 0.0)
            or np.any(
                position > self.environment_size
            )
        )

    # =====================================================
    # COLLISION CHECK
    # =====================================================

    def _check_collision(self):

        for obstacle in self.obstacles:

            if obstacle.check_collision(
                self.auv_position,
                self.auv_radius
            ):
                return True

        return False

    # =====================================================
    # GET NEARBY OBSTACLES
    # =====================================================

    def _get_nearby_obstacles(self):

        obstacle_information = []

        for obstacle in self.obstacles:

            relative_position = (
                obstacle.position
                - self.auv_position
            )

            distance = np.linalg.norm(
                relative_position
            )

            obstacle_information.append(
                (
                    distance,
                    relative_position
                )
            )

        obstacle_information.sort(
            key=lambda item: item[0]
        )

        nearby_obstacles = []

        nearest_obstacles = (
            obstacle_information[
                :self.max_nearby_obstacles
            ]
        )

        for distance, relative_position in nearest_obstacles:

            nearby_obstacles.extend(
                [
                    relative_position[0],
                    relative_position[1],
                    relative_position[2],
                    distance
                ]
            )

        required_size = (
            self.max_nearby_obstacles * 4
        )

        while len(nearby_obstacles) < required_size:

            nearby_obstacles.append(0.0)

        return np.array(
            nearby_obstacles,
            dtype=np.float32
        )

    # =====================================================
    # SAFETY VIOLATION CHECK
    # =====================================================

    def _check_safety_violation(self):

        for obstacle in self.obstacles:

            distance = np.linalg.norm(
                self.auv_position
                - obstacle.position
            )

            safe_distance = (
                self.auv_radius
                + obstacle.radius
                + self.safety_distance
            )

            if distance < safe_distance:

                return True

        return False

    # =====================================================
    # GOAL CHECK
    # =====================================================

    def _check_goal_reached(self):

        distance_to_goal = np.linalg.norm(
            self.auv_position
            - self.goal_position
        )

        return bool(
            distance_to_goal
            <= self.goal_radius
        )

    # =====================================================
    # RANDOM START POSITION
    # =====================================================

    def _generate_random_start_position(self):

        for _ in range(100):

            position = self.np_random.uniform(
                low=np.array(
                    [1.0, 1.0, 1.0]
                ),
                high=self.environment_size - 1.0
            ).astype(np.float32)

            # Keep the AUV sufficiently away from the goal
            distance_to_goal = np.linalg.norm(
                position
                - self.goal_position
            )

            if distance_to_goal < 5.0:
                continue

            valid_position = True

            # Check all obstacles
            for obstacle in self.obstacles:

                distance = np.linalg.norm(
                    position
                    - obstacle.position
                )

                minimum_safe_distance = (
                    self.auv_radius
                    + obstacle.radius
                    + self.safety_distance
                )

                if distance < minimum_safe_distance:

                    valid_position = False
                    break

            if valid_position:

                return position

        # Fallback position
        return self.start_position.copy()

    # =====================================================
    # RESET ENVIRONMENT
    # =====================================================

    def reset(self, seed=None, options=None):

        # Initialize Gymnasium random generator
        super().reset(seed=seed)

        # Generate random valid starting position
        self.auv_position = (
            self._generate_random_start_position()
        )

        # Reset step count
        self.current_step = 0

        # Reset energy
        self.energy_model.reset()

        # Get initial observation
        observation = self._get_observation()

        # Information dictionary
        info = {

            "remaining_energy": float(
                self.energy_model.remaining_energy
            ),

            "step": int(
                self.current_step
            ),

            "start_position": self.auv_position.copy()
        }

        # Gymnasium requires exactly:
        # (observation, info)
        return observation, info

    # =====================================================
    # ENVIRONMENT STEP
    # =====================================================

    def step(self, action):

        # Process action
        action = np.array(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        # Increase step count
        self.current_step += 1

        # Store previous position
        previous_position = (
            self.auv_position.copy()
        )

        # Consume energy
        energy_consumed = (
            self.energy_model.consume_energy(
                action
            )
        )

        energy_depleted = bool(
            self.energy_model.is_depleted()
        )

        # Move AUV
        new_position = self.dynamics.move(
            self.auv_position,
            action
        )

        # Check boundary before clipping
        hit_boundary = bool(
            self._is_out_of_bounds(
                new_position
            )
        )

        # Keep position within valid environment range
        self.auv_position = np.clip(
            new_position,
            [0.0, 0.0, 0.0],
            self.environment_size
        ).astype(np.float32)

        # Check collision
        collision = bool(
            self._check_collision()
        )

        # Check safety violation
        safety_violation = bool(
            self._check_safety_violation()
        )

        # Check goal
        reached_goal = bool(
            self._check_goal_reached()
        )

        # Distance to goal
        distance_to_goal = float(
            np.linalg.norm(
                self.auv_position
                - self.goal_position
            )
        )

        # =================================================
        # TERMINATION
        # =================================================

        terminated = bool(
            collision
            or reached_goal
            or energy_depleted
            or hit_boundary
        )

        truncated = bool(
            self.current_step
            >= self.max_steps
        )

        # =================================================
        # REWARD
        # =================================================

        reward = (
            self.reward_calculator.calculate(
                previous_position=previous_position,
                current_position=self.auv_position,
                goal_position=self.goal_position,
                collision=collision,
                reached_goal=reached_goal,
                safety_violation=safety_violation,
                energy_consumed=energy_consumed,
                hit_boundary=hit_boundary
            )
        )

        reward = float(
            reward
        )

        # New observation
        observation = (
            self._get_observation()
        )

        # =================================================
        # INFO
        # =================================================

        info = {

            "collision": bool(
                collision
            ),

            "safety_violation": bool(
                safety_violation
            ),

            "hit_boundary": bool(
                hit_boundary
            ),

            "reached_goal": bool(
                reached_goal
            ),

            "energy_consumed": float(
                energy_consumed
            ),

            "remaining_energy": float(
                self.energy_model.remaining_energy
            ),

            "energy_depleted": bool(
                energy_depleted
            ),

            "distance_to_goal": float(
                distance_to_goal
            ),

            "step": int(
                self.current_step
            )
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )