import numpy as np

from environment.auv_env import AUVEnvironment


def main():

    # -------------------------------------------------
    # Create the AUV environment
    # -------------------------------------------------

    env = AUVEnvironment()

    # Reset the environment
    observation, info = env.reset()

    print("\n" + "=" * 60)
    print("AUV SAFETY AND ENERGY TEST")
    print("=" * 60)

    print("\nInitial AUV position:")
    print(env.auv_position)

    print("\nInitial energy:")
    print(env.energy_model.remaining_energy)

    print("\nObstacle positions:")

    for index, obstacle in enumerate(env.obstacles):

        print(
            f"Obstacle {index + 1}: "
            f"{obstacle.position}"
        )

    # -------------------------------------------------
    # Select the first obstacle as the target
    #
    # We deliberately move toward it so that we can
    # test safety detection while consuming energy.
    # -------------------------------------------------

    target_obstacle = env.obstacles[0]

    print("\nTarget obstacle:")
    print(target_obstacle.position)

    print("\nStarting controlled movement...\n")

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # -------------------------------------------------
        # Calculate direction from the AUV
        # toward the target obstacle
        # -------------------------------------------------

        direction = (
            target_obstacle.position
            - env.auv_position
        )

        # Calculate distance
        distance = np.linalg.norm(direction)

        # Normalize the direction.
        #
        # This gives us a movement action with a
        # magnitude of approximately 1.
        # -------------------------------------------------

        if distance > 0:

            action = direction / distance

        else:

            action = np.zeros(3)

        # -------------------------------------------------
        # Perform one environment step
        #
        # This internally:
        # 1. Consumes energy
        # 2. Moves the AUV
        # 3. Checks safety
        # 4. Checks collision
        # 5. Calculates reward
        # -------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        # -------------------------------------------------
        # Display results
        # -------------------------------------------------

        print(f"Step {info['step']}")

        print(
            "AUV position:",
            env.auv_position
        )

        print(
            "Distance to target obstacle:",
            np.linalg.norm(
                env.auv_position
                - target_obstacle.position
            )
        )

        print(
            "Safety violation:",
            info["safety_violation"]
        )

        print(
            "Collision:",
            info["collision"]
        )

        # -------------------------------------------------
        # Energy information
        # -------------------------------------------------

        print(
            "Energy consumed:",
            info["energy_consumed"]
        )

        print(
            "Remaining energy:",
            info["remaining_energy"]
        )

        print(
            "Energy depleted:",
            info["energy_depleted"]
        )

        print(
            "Reward:",
            reward
        )

        print("-" * 60)

        # -------------------------------------------------
        # Stop after collision
        # -------------------------------------------------

        if info["collision"]:

            print("\nResult: Collision detected.")

            break

        # -------------------------------------------------
        # Stop after entering the safety zone
        # -------------------------------------------------

        if info["safety_violation"]:

            print(
                "\nResult: Safety zone entered successfully."
            )

            break


if __name__ == "__main__":

    main()