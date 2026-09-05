import os
import sys
import numpy as np

# =================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )

# =================================================
# IMPORTS
# =================================================

from stable_baselines3 import PPO

from environment.auv_env import AUVEnvironment


def main():

    # =================================================
    # CREATE ENVIRONMENT
    # =================================================

    print("\nLoading environment...")

    env = AUVEnvironment()

    # =================================================
    # LOAD MODEL
    # =================================================

    print(
        "Loading random-start PPO model..."
    )

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "auv_ppo_random_start"
    )

    model = PPO.load(
        model_path
    )

    print(
        "Model loaded successfully!"
    )

    # =================================================
    # EVALUATION SETTINGS
    # =================================================

    number_of_episodes = 20

    successful_episodes = 0

    collision_count = 0

    safety_violation_count = 0

    boundary_hit_count = 0

    total_rewards = []

    remaining_energies = []

    episode_lengths = []

    # =================================================
    # START EVALUATION
    # =================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EVALUATING RANDOM START POSITIONS"
    )

    print(
        "=" * 60
    )

    # =================================================
    # RUN MULTIPLE EPISODES
    # =================================================

    for episode in range(
        1,
        number_of_episodes + 1
    ):

        observation, info = env.reset()

        start_position = (
            env.auv_position.copy()
        )

        total_reward = 0.0

        terminated = False

        truncated = False

        final_info = {}

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
            ) = env.step(
                action
            )

            total_reward += reward

            final_info = info

        # =============================================
        # COLLECT RESULTS
        # =============================================

        reached_goal = (
            final_info["reached_goal"]
        )

        collision = (
            final_info["collision"]
        )

        safety_violation = (
            final_info["safety_violation"]
        )

        hit_boundary = (
            final_info["hit_boundary"]
        )

        remaining_energy = (
            final_info["remaining_energy"]
        )

        steps = (
            final_info["step"]
        )

        # =============================================
        # UPDATE STATISTICS
        # =============================================

        if reached_goal:

            successful_episodes += 1

        if collision:

            collision_count += 1

        if safety_violation:

            safety_violation_count += 1

        if hit_boundary:

            boundary_hit_count += 1

        total_rewards.append(
            total_reward
        )

        remaining_energies.append(
            remaining_energy
        )

        episode_lengths.append(
            steps
        )

        # =============================================
        # PRINT EPISODE RESULT
        # =============================================

        print(
            f"\nEpisode {episode}"
        )

        print(
            f"Start position: "
            f"{np.round(start_position, 2)}"
        )

        print(
            f"Final position: "
            f"{np.round(env.auv_position, 2)}"
        )

        print(
            f"Reached goal: "
            f"{reached_goal}"
        )

        print(
            f"Collision: "
            f"{collision}"
        )

        print(
            f"Safety violation: "
            f"{safety_violation}"
        )

        print(
            f"Hit boundary: "
            f"{hit_boundary}"
        )

        print(
            f"Remaining energy: "
            f"{remaining_energy:.2f}"
        )

        print(
            f"Steps: "
            f"{steps}"
        )

        print(
            f"Total reward: "
            f"{total_reward:.2f}"
        )

    # =================================================
    # FINAL RESULTS
    # =================================================

    success_rate = (
        successful_episodes
        / number_of_episodes
        * 100
    )

    collision_rate = (
        collision_count
        / number_of_episodes
        * 100
    )

    boundary_rate = (
        boundary_hit_count
        / number_of_episodes
        * 100
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL EVALUATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Total episodes: "
        f"{number_of_episodes}"
    )

    print(
        f"Successful episodes: "
        f"{successful_episodes}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.2f}%"
    )

    print(
        f"Collisions: "
        f"{collision_count}"
    )

    print(
        f"Collision rate: "
        f"{collision_rate:.2f}%"
    )

    print(
        f"Safety violations: "
        f"{safety_violation_count}"
    )

    print(
        f"Boundary hits: "
        f"{boundary_hit_count}"
    )

    print(
        f"Boundary hit rate: "
        f"{boundary_rate:.2f}%"
    )

    print(
        f"Average remaining energy: "
        f"{np.mean(remaining_energies):.2f}"
    )

    print(
        f"Average episode reward: "
        f"{np.mean(total_rewards):.2f}"
    )

    print(
        f"Average episode length: "
        f"{np.mean(episode_lengths):.2f}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()