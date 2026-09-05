import numpy as np

from stable_baselines3 import PPO

from environment.auv_env import AUVEnvironment


def main():

    print("\n" + "=" * 60)
    print("TRAINED PPO AUV TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Create the environment
    # -------------------------------------------------

    env = AUVEnvironment()

    # -------------------------------------------------
    # Load the trained PPO model
    # -------------------------------------------------

    model = PPO.load(
        "models/auv_ppo_model"
    )

    # -------------------------------------------------
    # Reset the environment
    # -------------------------------------------------

    observation, info = env.reset()

    print("\nInitial AUV position:")
    print(env.auv_position)

    print("\nGoal position:")
    print(env.goal_position)

    print("\nInitial energy:")
    print(env.energy_model.remaining_energy)

    print("\nStarting trained PPO agent...\n")

    terminated = False
    truncated = False

    total_reward = 0.0

    # -------------------------------------------------
    # Run one complete episode
    # -------------------------------------------------

    while not terminated and not truncated:

        # PPO selects an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # Perform the action
        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += reward

        # Print episode information
        print(f"Step: {info['step']}")

        print(
            "Action:",
            action
        )

        print(
            "AUV position:",
            env.auv_position
        )

        print(
            "Remaining energy:",
            info["remaining_energy"]
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
            "Reward:",
            reward
        )

        print("-" * 60)

    # -------------------------------------------------
    # Print final results
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("PPO AUV NAVIGATION RESULT")
    print("=" * 60)

    print(
        "Final AUV position:",
        env.auv_position
    )

    print(
        "Goal position:",
        env.goal_position
    )

    print(
        "Steps taken:",
        info["step"]
    )

    print(
        "Total reward:",
        total_reward
    )

    print(
        "Remaining energy:",
        info["remaining_energy"]
    )

    print(
        "Reached goal:",
        info["reached_goal"]
    )

    print(
        "Collision:",
        info["collision"]
    )

    print(
        "Energy depleted:",
        info["energy_depleted"]
    )

    # -------------------------------------------------
    # Final result
    # -------------------------------------------------

    if info["reached_goal"]:

        print(
            "\nRESULT: GOAL REACHED SUCCESSFULLY!"
        )

    elif info["collision"]:

        print(
            "\nRESULT: COLLISION OCCURRED."
        )

    elif info["energy_depleted"]:

        print(
            "\nRESULT: ENERGY DEPLETED."
        )

    else:

        print(
            "\nRESULT: MAXIMUM STEPS REACHED."
        )


if __name__ == "__main__":
    main()