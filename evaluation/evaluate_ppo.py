import os
import sys
import numpy as np

from stable_baselines3 import PPO


# Get the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add project root to Python path
sys.path.insert(
    0,
    PROJECT_ROOT
)

from environment.auv_env import AUVEnvironment


def main():

    # =============================================
    # LOAD ENVIRONMENT
    # =============================================

    env = AUVEnvironment()

    # =============================================
    # LOAD TRAINED MODEL
    # =============================================

    print("\nLoading trained PPO model...")

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "auv_ppo_model"
    )

    model = PPO.load(
        model_path
    )

    print(
        "Model loaded successfully!"
    )

    # =============================================
    # EVALUATION SETTINGS
    # =============================================

    num_episodes = 20

    successful_episodes = 0
    collision_episodes = 0
    total_rewards = []

    # =============================================
    # RUN EVALUATION
    # =============================================

    for episode in range(num_episodes):

        observation, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0.0

        while not (
            terminated
            or truncated
        ):

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

        total_rewards.append(
            total_reward
        )

        if info["reached_goal"]:

            successful_episodes += 1

        if info["collision"]:

            collision_episodes += 1

        print(
            f"\nEpisode {episode + 1}"
        )

        print(
            f"Reward: {total_reward:.2f}"
        )

        print(
            f"Steps: {info['step']}"
        )

        print(
            f"Goal Reached: {info['reached_goal']}"
        )

        print(
            f"Collision: {info['collision']}"
        )

        print(
            f"Safety Violation: "
            f"{info['safety_violation']}"
        )

        print(
            f"Remaining Energy: "
            f"{info['remaining_energy']:.2f}"
        )

    # =============================================
    # FINAL RESULTS
    # =============================================

    success_rate = (
        successful_episodes
        / num_episodes
        * 100
    )

    collision_rate = (
        collision_episodes
        / num_episodes
        * 100
    )

    print("\n")
    print("=" * 50)
    print("PPO EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Total Episodes: {num_episodes}"
    )

    print(
        f"Successful Episodes: "
        f"{successful_episodes}"
    )

    print(
        f"Success Rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Collision Rate: "
        f"{collision_rate:.2f}%"
    )

    print(
        f"Average Reward: "
        f"{np.mean(total_rewards):.2f}"
    )

    print(
        f"Average Steps: "
        f"{np.mean([info['step']]):.2f}"
    )


if __name__ == "__main__":
    main()