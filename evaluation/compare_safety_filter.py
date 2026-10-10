
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from environment.auv_env import AUVEnvironment
from environment.safety_filter import SafetyFilter

MODEL_PATH = "models/auv_ppo_improved_safety"
EPISODES = 20


def evaluate(use_filter):
    env = AUVEnvironment()
    model = PPO.load(MODEL_PATH, device="cpu")

    successes = 0
    collisions = 0
    safety_violations = 0
    boundaries = 0
    energy_depletions = 0
    timeouts = 0
    total_steps = 0
    intervention_count = 0

    label = "FILTERED" if use_filter else "ORIGINAL"

    for episode in range(EPISODES):
        obs, info = env.reset(seed=1000 + episode)

        safety_filter = SafetyFilter(
            obstacles=env.obstacles,
            auv_radius=env.auv_radius,
            speed=env.dynamics.speed,
            safety_distance=env.safety_distance,
            environment_size=env.environment_size,
        )

        terminated = False
        truncated = False
        episode_steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            if use_filter:
                filtered_action = safety_filter.filter_action(
                    env.auv_position, action, goal_position=env.goal_position,
                )

                if not np.allclose(action, filtered_action):
                    intervention_count += 1

                action = filtered_action

            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1

        success = bool(info.get("reached_goal", False))
        collision = bool(info.get("collision", False))
        violation = bool(info.get("safety_violation", False))
        boundary = bool(info.get("hit_boundary", False))
        depleted = bool(info.get("energy_depleted", False))

        successes += int(success)
        collisions += int(collision)
        safety_violations += int(violation)
        boundaries += int(boundary)
        energy_depletions += int(depleted)
        timeouts += int(truncated and not success)
        total_steps += episode_steps

        if not success:
            print(
                f"{label} episode {episode + 1}: "
                f"success={success}, collision={collision}, "
                f"safety_violation={violation}, boundary={boundary}, "
                f"energy_depleted={depleted}, timeout={truncated}, "
                f"steps={episode_steps}, "
                f"distance_to_goal={info.get('distance_to_goal', 'unknown')}"
            )

    env.close()

    return {
        "success_rate": 100 * successes / EPISODES,
        "collisions": collisions,
        "safety_violations": safety_violations,
        "boundary_hits": boundaries,
        "energy_depletions": energy_depletions,
        "timeouts": timeouts,
        "average_steps": total_steps / EPISODES,
        "filter_interventions": intervention_count,
    }


if __name__ == "__main__":
    print("\nEvaluating ORIGINAL PPO...")
    original = evaluate(use_filter=False)

    print("\nEvaluating PPO + SAFETY FILTER...")
    filtered = evaluate(use_filter=True)

    print("\n========== COMPARISON ==========")

    for name, results in (
        ("Original PPO", original),
        ("PPO + Safety Filter", filtered),
    ):
        print(f"\n{name}")
        for metric, value in results.items():
            print(f"{metric}: {value}")
