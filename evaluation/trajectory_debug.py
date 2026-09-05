import os
import sys
import numpy as np

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from stable_baselines3 import PPO
from environment.auv_env import AUVEnvironment


def main():

    print("\nLoading environment...")

    env = AUVEnvironment()

    print("Loading trained PPO model...")

    model = PPO.load(
        "models/auv_ppo_model"
    )

    print("Model loaded successfully!")

    observation, info = env.reset()

    print("\nINITIAL POSITION:")

    print(env.auv_position)

    print("\nGOAL POSITION:")

    print(env.goal_position)

    print(
        "\n"
        + "=" * 60
    )

    print("AUV TRAJECTORY")

    print(
        "=" * 60
    )

    total_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        print(
            f"Step {info['step']:3d} | "
            f"Action: {np.round(action, 3)} | "
            f"Position: {np.round(env.auv_position, 3)} | "
            f"Distance: {info['distance_to_goal']:.3f} | "
            f"Reward: {reward:.3f}"
        )

    print(
        "\n"
        + "=" * 60
    )

    print("FINAL RESULT")

    print(
        "=" * 60
    )

    print(
        f"Final position: {env.auv_position}"
    )

    print(
        f"Goal position: {env.goal_position}"
    )

    print(
        f"Reached goal: {info['reached_goal']}"
    )

    print(
        f"Collision: {info['collision']}"
    )

    print(
        f"Safety violation: "
        f"{info['safety_violation']}"
    )

    print(
        f"Hit boundary: "
        f"{info['hit_boundary']}"
    )

    print(
        f"Remaining energy: "
        f"{info['remaining_energy']:.2f}"
    )

    print(
        f"Total reward: {total_reward:.3f}"
    )


if __name__ == "__main__":
    main()