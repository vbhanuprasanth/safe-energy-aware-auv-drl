
import numpy as np


class Obstacle:
    """Represents a spherical obstacle in the 3D environment."""

    def __init__(self, position, radius=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.radius = float(radius)

    def check_collision(self, point, auv_radius=0.5):
        """Check collision at a single position."""
        point = np.asarray(point, dtype=np.float32)
        distance = np.linalg.norm(point - self.position)
        return bool(distance <= self.radius + auv_radius)

    def check_path_collision(
        self, start, end, auv_radius=0.5
    ):
        """Check whether a straight movement segment intersects the sphere."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        segment = end - start
        length_squared = float(np.dot(segment, segment))

        if length_squared == 0.0:
            return self.check_collision(start, auv_radius)

        # Closest point on the movement segment to the obstacle centre.
        t = float(
            np.dot(self.position - start, segment) / length_squared
        )
        t = float(np.clip(t, 0.0, 1.0))
        closest_point = start + t * segment

        distance = np.linalg.norm(closest_point - self.position)
        return bool(distance <= self.radius + auv_radius)
