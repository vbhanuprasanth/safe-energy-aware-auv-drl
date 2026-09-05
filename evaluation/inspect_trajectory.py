import os
import sys
import numpy as np

from stable_baselines3 import PPO


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from environment.auv_env import AUVEnvironment


def main():

    env = AUVEnvironment()

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "auv_ppo_model"
    )

    model = PPO.load(
        model_path
    )

    observation, info = env.reset()

    print("\nINITIAL POSITION:")
    print(env.auv_position)

    print("\nGOAL POSITION:")
    print(env.goal_position)

    print("\n" + "=" * 60)
    print("AUV TRAJECTORY")
    print("=" * 60)

    terminated = False
    truncated = False

    total_reward = 0.0

    while not (terminated or truncated):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += reward

        print(
            f"Step {info['step']:3d} | "
            f"Action: {np.round(action, 3)} | "
            f"Position: {np.round(env.auv_position, 3)} | "
            f"Distance: {info['distance_to_goal']:.3f} | "
            f"Reward: {reward:.3f}"
        )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("Final position:", env.auv_position)
    print("Goal position:", env.goal_position)
    print("Reached goal:", info["reached_goal"])
    print("Collision:", info["collision"])
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()