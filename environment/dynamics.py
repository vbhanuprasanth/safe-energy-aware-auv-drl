
import numpy as np


class AUVDynamics:
    """Basic AUV movement with optional water-current disturbance."""

    def __init__(self, speed=1.0, current_strength=0.0, seed=None):
        self.speed = float(speed)
        self.current_strength = float(current_strength)
        self.rng = np.random.default_rng(seed)

        if self.current_strength < 0.0:
            raise ValueError("current_strength must be non-negative")

    def move(self, position, action):
        """
        Move the AUV using its action and an external current.

        position: Current position [x, y, z]
        action: Three PPO control values
        current_strength: Maximum magnitude of the current vector
        """
        position = np.asarray(position, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        if position.shape != (3,) or action.shape != (3,):
            raise ValueError("position and action must each have 3 values")

        current = np.zeros(3, dtype=np.float32)

        if self.current_strength > 0.0:
            # Random current direction, with magnitude up to the configured limit.
            direction = self.rng.normal(size=3)
            norm = np.linalg.norm(direction)

            if norm > 0.0:
                direction = direction / norm
                magnitude = self.rng.uniform(0.0, self.current_strength)
                current = (direction * magnitude).astype(np.float32)

        new_position = position + action * self.speed + current

        return new_position.astype(np.float32)
