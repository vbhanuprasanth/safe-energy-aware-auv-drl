
import itertools
import numpy as np


class SafetyFilter:
    """Filter unsafe actions while favoring progress toward the goal."""

    def __init__(
        self,
        obstacles,
        auv_radius=0.5,
        speed=1.0,
        safety_distance=2.0,
        environment_size=None,
    ):
        self.obstacles = obstacles
        self.auv_radius = float(auv_radius)
        self.speed = float(speed)
        self.safety_distance = float(safety_distance)

        self.environment_size = (
            None
            if environment_size is None
            else np.asarray(environment_size, dtype=np.float32)
        )

    def is_safe(self, position, action):
        position = np.asarray(position, dtype=np.float32)
        action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )

        proposed_position = position + action * self.speed
        segment = proposed_position - position

        # Reject movements outside the environment.
        if self.environment_size is not None:
            if (
                np.any(proposed_position < 0.0)
                or np.any(proposed_position > self.environment_size)
            ):
                return False

        length_squared = float(np.dot(segment, segment))

        # Check the complete movement segment against every obstacle.
        for obstacle in self.obstacles:
            required_distance = (
                self.auv_radius
                + float(obstacle.radius)
                + self.safety_distance
            )

            if length_squared == 0.0:
                closest_point = position
            else:
                t = float(
                    np.dot(obstacle.position - position, segment)
                    / length_squared
                )
                t = float(np.clip(t, 0.0, 1.0))
                closest_point = position + t * segment

            distance = float(
                np.linalg.norm(closest_point - obstacle.position)
            )

            if distance < required_distance:
                return False

        return True

    def filter_action(self, position, action, goal_position=None):
        position = np.asarray(position, dtype=np.float32)
        action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )

        # Keep the PPO action if it already satisfies all checks.
        if self.is_safe(position, action):
            return action

        candidates = []

        # Try scaled versions of the PPO action.
        for scale in (0.75, 0.5, 0.25, 0.0):
            candidates.append(action * scale)

        # Try alternative directions.
        for direction in itertools.product((-1, 0, 1), repeat=3):
            if direction == (0, 0, 0):
                continue

            candidate = np.asarray(direction, dtype=np.float32)
            candidate /= np.linalg.norm(candidate)
            candidates.append(candidate)

        safe_candidates = [
            candidate
            for candidate in candidates
            if self.is_safe(position, candidate)
        ]

        if not safe_candidates:
            # No safe candidate was found. Zero is only a fallback.
            return np.zeros(3, dtype=np.float32)

        if goal_position is None:
            # Prefer the safe action closest to the PPO proposal.
            return min(
                safe_candidates,
                key=lambda candidate: float(
                    np.linalg.norm(candidate - action)
                ),
            )

        goal_position = np.asarray(goal_position, dtype=np.float32)
        current_distance = float(
            np.linalg.norm(goal_position - position)
        )

        def score(candidate):
            next_position = position + candidate * self.speed
            next_distance = float(
                np.linalg.norm(goal_position - next_position)
            )

            progress = current_distance - next_distance
            policy_similarity = -float(
                np.linalg.norm(candidate - action)
            )

            return progress + 0.1 * policy_similarity

        return max(safe_candidates, key=score)
