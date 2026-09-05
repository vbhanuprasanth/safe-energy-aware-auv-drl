import os
import sys

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
from stable_baselines3.common.env_checker import check_env

from environment.auv_env import AUVEnvironment


def main():

    # =================================================
    # CREATE AUV ENVIRONMENT
    # =================================================

    print("\nCreating AUV environment...")

    env = AUVEnvironment()

    # =================================================
    # CHECK ENVIRONMENT
    # =================================================

    print("\nChecking environment...")

    check_env(
        env,
        warn=True
    )

    print(
        "Environment check passed successfully!"
    )

    # =================================================
    # CREATE PPO MODEL
    # =================================================

    print("\nCreating improved PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        gamma=0.99,

        gae_lambda=0.95,

        ent_coef=0.01,

        seed=42
    )

    # =================================================
    # TRAIN PPO
    # =================================================

    print(
        "\nStarting improved PPO training "
        "with random starting positions...\n"
    )

    model.learn(
        total_timesteps=300000
    )

    # =================================================
    # CREATE MODELS DIRECTORY
    # =================================================

    models_directory = os.path.join(
        PROJECT_ROOT,
        "models"
    )

    os.makedirs(
        models_directory,
        exist_ok=True
    )

    # =================================================
    # SAVE NEW MODEL
    # =================================================

    model_path = os.path.join(
        models_directory,
        "auv_ppo_improved_safety"
    )

    model.save(
        model_path
    )

    # =================================================
    # FINISH
    # =================================================

    print(
        "\nTraining completed successfully!"
    )

    print(
        f"Improved model saved in "
        f"{model_path}.zip"
    )

    print(
        "\nPrevious models have not been overwritten."
    )


if __name__ == "__main__":
    main()