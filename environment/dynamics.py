import numpy as np


class AUVDynamics:
    """
    Handles the basic movement of the AUV inside the environment.
    """

    def __init__(self, speed=1.0):
        self.speed = speed

    def move(self, position, action):
        """
        Move the AUV according to the selected action.

        position: Current AUV position [x, y, z]
        action: Movement direction [dx, dy, dz]
        """

        position = np.array(position, dtype=np.float32)
        action = np.array(action, dtype=np.float32)

        new_position = position + action * self.speed

        return new_position