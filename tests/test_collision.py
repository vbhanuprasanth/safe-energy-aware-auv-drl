import numpy as np

from environment.auv_env import AUVEnvironment


def main():

    # Create environment
    env = AUVEnvironment()

    # Reset environment
    observation, info = env.reset()

    print("\n" + "=" * 60)
    print("AUV COLLISION TEST")
    print("=" * 60)

    print("\nInitial AUV position:")
    print(env.auv_position)

    print("\nInitial energy:")
    print(env.energy_model.remaining_energy)

    # Select the first obstacle
    target_obstacle = env.obstacles[0]

    print("\nTarget obstacle:")
    print(target_obstacle.position)

    print(
        "\nStarting controlled movement "
        "toward the obstacle...\n"
    )

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Calculate direction toward obstacle
        direction = (
            target_obstacle.position
            - env.auv_position
        )

        # Normalize direction
        distance = np.linalg.norm(direction)

        if distance > 0:
            action = direction / distance
        else:
            action = np.zeros(3)

        # Perform movement
        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        print(f"Step {info['step']}")

        print(
            "AUV position:",
            env.auv_position
        )

        print(
            "Distance to obstacle:",
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

        print(
            "Energy consumed:",
            info["energy_consumed"]
        )

        print(
            "Remaining energy:",
            info["remaining_energy"]
        )

        print(
            "Reward:",
            reward
        )

        print("-" * 60)

        # Check result
        if info["collision"]:

            print("\n" + "=" * 60)
            print("RESULT: COLLISION DETECTED SUCCESSFULLY!")
            print("Episode terminated:", terminated)
            print("=" * 60)

            break


if __name__ == "__main__":
    main()