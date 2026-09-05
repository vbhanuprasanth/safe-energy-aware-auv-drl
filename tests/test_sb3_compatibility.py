from stable_baselines3.common.env_checker import check_env

from environment.auv_env import AUVEnvironment


def test_sb3_environment_compatibility():

    env = AUVEnvironment()

    check_env(
        env,
        warn=True
    )