import numpy as np


class RewardCalculator:
    """
    Reward system for safe and energy-aware
    AUV navigation.
    """

    def __init__(self):

        # ==========================================
        # GOAL REWARD
        # ==========================================

        # Large reward for successfully
        # reaching the goal.

        self.goal_reward = 250.0

        # ==========================================
        # COLLISION PENALTY
        # ==========================================

        # Collision should be one of the
        # worst possible outcomes.

        self.collision_penalty = -250.0

        # ==========================================
        # SAFETY VIOLATION PENALTY
        # ==========================================

        # Entering the obstacle safety zone
        # should be strongly discouraged.

        self.safety_penalty = -25.0

        # ==========================================
        # BOUNDARY PENALTY
        # ==========================================

        self.boundary_penalty = -100.0

        # ==========================================
        # PROGRESS REWARD
        # ==========================================

        # Reward movement toward the goal.

        self.progress_scale = 20.0

        # ==========================================
        # ENERGY PENALTY
        # ==========================================

        self.energy_penalty_scale = 1.0

        # ==========================================
        # STEP PENALTY
        # ==========================================

        # Encourage efficient navigation.

        self.step_penalty = -0.2

    def calculate(
        self,
        previous_position,
        current_position,
        goal_position,
        collision,
        reached_goal,
        safety_violation,
        energy_consumed,
        hit_boundary
    ):
        """
        Calculate the reward for one AUV step.
        """

        # ==========================================
        # DISTANCE BEFORE MOVEMENT
        # ==========================================

        previous_distance = np.linalg.norm(
            previous_position - goal_position
        )

        # ==========================================
        # DISTANCE AFTER MOVEMENT
        # ==========================================

        current_distance = np.linalg.norm(
            current_position - goal_position
        )

        # ==========================================
        # PROGRESS TOWARD GOAL
        # ==========================================

        distance_progress = (
            previous_distance
            - current_distance
        )

        reward = (
            distance_progress
            * self.progress_scale
        )

        # ==========================================
        # STEP PENALTY
        # ==========================================

        reward += self.step_penalty

        # ==========================================
        # ENERGY PENALTY
        # ==========================================

        reward -= (
            energy_consumed
            * self.energy_penalty_scale
        )

        # ==========================================
        # SAFETY PENALTY
        # ==========================================

        if safety_violation:

            reward += self.safety_penalty

        # ==========================================
        # BOUNDARY PENALTY
        # ==========================================

        if hit_boundary:

            reward += self.boundary_penalty

        # ==========================================
        # COLLISION PENALTY
        # ==========================================

        if collision:

            reward += self.collision_penalty

        # ==========================================
        # GOAL REWARD
        # ==========================================

        if reached_goal:

            reward += self.goal_reward

        return float(reward)