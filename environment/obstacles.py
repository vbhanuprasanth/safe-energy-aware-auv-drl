import numpy as np


class Obstacle:
    """
    Represents a spherical obstacle in the 3D environment.
    """

    def __init__(self, position, radius=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius

    def check_collision(self, point, auv_radius=0.5):
        """
        Check whether the AUV collides with this obstacle.
        """

        point = np.array(point, dtype=np.float32)

        distance = np.linalg.norm(point - self.position)

        return distance <= self.radius + auv_radius