import numpy as np
import matplotlib.pyplot as plt


class AUVVisualizer:
    """
    Visualizes the AUV navigation environment in 3D.
    """

    def __init__(self, environment):
        self.environment = environment

    def plot_environment(self, path=None):
        """
        Display the AUV environment, obstacles, start, goal,
        current AUV position, and optional navigation path.
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        env_size = self.environment.environment_size

        # ---------------------------------------------
        # Plot start position
        # ---------------------------------------------

        start = self.environment.start_position

        ax.scatter(
            start[0],
            start[1],
            start[2],
            marker="o",
            s=100,
            label="Start"
        )

        # ---------------------------------------------
        # Plot current AUV position
        # ---------------------------------------------

        auv = self.environment.auv_position

        ax.scatter(
            auv[0],
            auv[1],
            auv[2],
            marker="^",
            s=100,
            label="AUV"
        )

        # ---------------------------------------------
        # Plot goal position
        # ---------------------------------------------

        goal = self.environment.goal_position

        ax.scatter(
            goal[0],
            goal[1],
            goal[2],
            marker="*",
            s=200,
            label="Goal"
        )

        # ---------------------------------------------
        # Plot obstacles
        # ---------------------------------------------

        for index, obstacle in enumerate(self.environment.obstacles):

            position = obstacle.position
            radius = obstacle.radius

            # Create spherical obstacle surface
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)

            x = position[0] + radius * np.outer(
                np.cos(u),
                np.sin(v)
            )

            y = position[1] + radius * np.outer(
                np.sin(u),
                np.sin(v)
            )

            z = position[2] + radius * np.outer(
                np.ones(np.size(u)),
                np.cos(v)
            )

            ax.plot_surface(
                x,
                y,
                z,
                alpha=0.5
            )

        # ---------------------------------------------
        # Plot AUV navigation path
        # ---------------------------------------------

        if path is not None and len(path) > 1:

            path = np.array(path)

            ax.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                linewidth=2,
                label="AUV Path"
            )

        # ---------------------------------------------
        # Configure axes
        # ---------------------------------------------

        ax.set_xlim(0, env_size[0])
        ax.set_ylim(0, env_size[1])
        ax.set_zlim(0, env_size[2])

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

        ax.set_title("3D AUV Navigation Environment")

        ax.legend()

        plt.show()