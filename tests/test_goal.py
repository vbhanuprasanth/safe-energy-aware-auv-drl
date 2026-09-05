import numpy as np

from environment.auv_env import AUVEnvironment


def main():

    # -------------------------------------------------
    # Create environment
    # -------------------------------------------------

    env = AUVEnvironment()

    # Reset environment
    observation, info = env.reset()

    print("\n" + "=" * 60)
    print("AUV GOAL REACHING TEST")
    print("=" * 60)

    print("\nInitial AUV position:")
    print(env.auv_position)

    print("\nGoal position:")
    print(env.goal_position)

    print("\nInitial energy:")
    print(env.energy_model.remaining_energy)

    print("\nStarting controlled movement toward goal...\n")

    terminated = False
    truncated = False

    # -------------------------------------------------
    # Move toward the goal
    # -------------------------------------------------

    while not terminated and not truncated:

        # Calculate direction from AUV to goal
        direction = (
            env.goal_position
            - env.auv_position
        )

        # Calculate distance to goal
        distance = np.linalg.norm(direction)

        # Normalize the direction
        if distance > 0:

            action = direction / distance

        else:

            action = np.zeros(3)

        # Perform one environment step
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
            "Distance to goal:",
            np.linalg.norm(
                env.auv_position
                - env.goal_position
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

        # -------------------------------------------------
        # Stop if goal is reached
        # -------------------------------------------------

        if info["reached_goal"]:

            print("\n" + "=" * 60)
            print("RESULT: GOAL REACHED SUCCESSFULLY!")
            print("=" * 60)

            break

        # Stop if collision occurs
        if info["collision"]:

            print("\n" + "=" * 60)
            print("RESULT: COLLISION OCCURRED!")
            print("=" * 60)

            break

        # Stop if energy is depleted
        if info["energy_depleted"]:

            print("\n" + "=" * 60)
            print("RESULT: ENERGY DEPLETED!")
            print("=" * 60)

            break


if __name__ == "__main__":
    main()