import os
import sys
import numpy as np

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from environment.auv_env import AUVEnvironment


def main():
    print("=" * 60)
    print("TEAM A PPO MODEL VERIFICATION")
    print("=" * 60)

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "auv_ppo_improved_safety.zip"
    )

    print("\nModel path:")
    print(model_path)

    print("\nLoading AUV environment...")
    env = AUVEnvironment()

    print("Environment created successfully.")

    print("\nLoading trained PPO model...")
    model = PPO.load(
        model_path,
        env=env
    )

    print("PPO model loaded successfully.")

    print("\nModel information:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    print("\nRunning inference test...")

    observation, info = env.reset(seed=42)

    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation: {observation}")

    for step in range(10):

        action, _states = model.predict(
            observation,
            deterministic=True
        )

        action = np.asarray(action)

        print(f"\nStep {step + 1}")
        print(f"Action: {action}")

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward}")
        print(f"Remaining energy: {info['remaining_energy']}")
        print(f"Distance to goal: {info['distance_to_goal']}")
        print(f"Collision: {info['collision']}")
        print(f"Safety violation: {info['safety_violation']}")
        print(f"Reached goal: {info['reached_goal']}")

        if terminated or truncated:
            print("\nEpisode ended during verification.")
            break

    print("\n" + "=" * 60)
    print("MODEL VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()