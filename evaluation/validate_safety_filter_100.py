
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from environment.auv_env import AUVEnvironment
from environment.safety_filter import SafetyFilter


MODEL_PATH = "models/auv_ppo_improved_safety"
EPISODES = 100
SEED_START = 2000


def evaluate(use_filter):
    env = AUVEnvironment()
    model = PPO.load(MODEL_PATH, device="cpu")

    results = {
        "successes": 0,
        "collisions": 0,
        "safety_violations": 0,
        "boundary_hits": 0,
        "energy_depletions": 0,
        "timeouts": 0,
        "total_steps": 0,
        "interventions": 0,
    }

    label = "FILTERED" if use_filter else "ORIGINAL"

    try:
        for episode in range(EPISODES):
            obs, info = env.reset(seed=SEED_START + episode)

            safety_filter = SafetyFilter(
                obstacles=env.obstacles,
                auv_radius=env.auv_radius,
                speed=env.dynamics.speed,
                safety_distance=env.safety_distance,
                environment_size=env.environment_size,
            )

            terminated = False
            truncated = False
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)

                if use_filter:
                    filtered_action = safety_filter.filter_action(
                        env.auv_position,
                        action,
                        goal_position=env.goal_position,
                    )

                    if not np.allclose(action, filtered_action):
                        results["interventions"] += 1

                    action = filtered_action

                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

            results["successes"] += int(
                bool(info.get("reached_goal", False))
            )
            results["collisions"] += int(
                bool(info.get("collision", False))
            )
            results["safety_violations"] += int(
                bool(info.get("safety_violation", False))
            )
            results["boundary_hits"] += int(
                bool(info.get("hit_boundary", False))
            )
            results["energy_depletions"] += int(
                bool(info.get("energy_depleted", False))
            )
            results["timeouts"] += int(
                bool(truncated and not info.get("reached_goal", False))
            )
            results["total_steps"] += steps

            if (episode + 1) % 10 == 0:
                print(f"{label}: completed {episode + 1}/{EPISODES}")

    finally:
        env.close()

    results["success_rate"] = (
        100.0 * results["successes"] / EPISODES
    )
    results["average_steps"] = (
        results["total_steps"] / EPISODES
    )

    return results


if __name__ == "__main__":
    print(f"Validating on {EPISODES} episodes.")
    print(f"Seeds: {SEED_START} through {SEED_START + EPISODES - 1}")

    print("\nEvaluating original PPO...")
    original = evaluate(use_filter=False)

    print("\nEvaluating PPO with safety filter...")
    filtered = evaluate(use_filter=True)

    print("\n========== 100-EPISODE VALIDATION ==========")

    for name, results in (
        ("Original PPO", original),
        ("PPO + Safety Filter", filtered),
    ):
        print(f"\n{name}")
        for metric, value in results.items():
            print(f"{metric}: {value}")
