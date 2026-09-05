import numpy as np


class EnergyModel:
    """
    Calculates energy consumption for AUV movement.
    """

    def __init__(self, initial_energy=100.0):
        """
        Initialize the AUV energy system.
        """

        self.initial_energy = initial_energy
        self.remaining_energy = initial_energy

    def reset(self):
        """
        Reset the energy at the beginning of an episode.
        """

        self.remaining_energy = self.initial_energy

    def calculate_consumption(self, action):
        """
        Calculate energy consumed by an action.

        Larger movements consume more energy.
        """

        action = np.array(action)

        # Magnitude of the movement
        action_magnitude = np.linalg.norm(action)

        # Energy consumption based on movement magnitude
        energy_consumed = 0.5 * action_magnitude

        return energy_consumed

    def consume_energy(self, action):
        """
        Consume energy according to the selected action.
        """

        energy_consumed = self.calculate_consumption(action)

        self.remaining_energy -= energy_consumed

        # Prevent negative energy
        self.remaining_energy = max(
            0.0,
            self.remaining_energy
        )

        return energy_consumed

    def is_depleted(self):
        """
        Check whether the AUV has run out of energy.
        """

        return self.remaining_energy <= 0.0