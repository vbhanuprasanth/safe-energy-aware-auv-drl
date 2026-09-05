import numpy as np

from environment.auv_env import AUVEnvironment


def test_observation_and_action_space():

    print("\n" + "=" * 60)
    print("TEST 1: OBSERVATION AND ACTION SPACE")
    print("=" * 60)

    env = AUVEnvironment()

    observation, info = env.reset()

    print("\nObservation shape:")
    print(observation.shape)

    print("\nExpected observation shape:")
    print(env.observation_space.shape)

    print("\nAction space:")
    print(env.action_space)

    print(
        "\nObservation valid:",
        env.observation_space.contains(observation)
    )


def test_goal_reaching():

    print("\n" + "=" * 60)
    print("TEST 2: GOAL REACHING")
    print("=" * 60)

    env = AUVEnvironment()

    observation, info = env.reset()

    print("\nGoal position:")
    print(env.goal_position)

    # Place the AUV close to the goal.
    # This tests whether the environment
    # correctly recognizes goal completion.

    env.auv_position = (
        env.goal_position.copy()
    )

    observation, reward, terminated, truncated, info = (
        env.step(
            np.zeros(3, dtype=np.float32)
        )
    )

    print("\nAUV position:")
    print(env.auv_position)

    print("\nReached goal:")
    print(info["reached_goal"])

    print("Terminated:")
    print(terminated)

    print("Reward:")
    print(reward)


def test_collision():

    print("\n" + "=" * 60)
    print("TEST 3: COLLISION DETECTION")
    print("=" * 60)

    env = AUVEnvironment()

    observation, info = env.reset()

    target_obstacle = env.obstacles[0]

    print("\nTarget obstacle:")
    print(target_obstacle.position)

    # Place the AUV directly at the
    # obstacle center.

    env.auv_position = (
        target_obstacle.position.copy()
    )

    observation, reward, terminated, truncated, info = (
        env.step(
            np.zeros(3, dtype=np.float32)
        )
    )

    print("\nAUV position:")
    print(env.auv_position)

    print("\nCollision:")
    print(info["collision"])

    print("Safety violation:")
    print(info["safety_violation"])

    print("Terminated:")
    print(terminated)

    print("Reward:")
    print(reward)


def test_energy_system():

    print("\n" + "=" * 60)
    print("TEST 4: ENERGY SYSTEM")
    print("=" * 60)

    env = AUVEnvironment()

    observation, info = env.reset()

    initial_energy = (
        env.energy_model.remaining_energy
    )

    print("\nInitial energy:")
    print(initial_energy)

    action = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("\nEnergy consumed:")
    print(info["energy_consumed"])

    print("Remaining energy:")
    print(info["remaining_energy"])

    print(
        "Energy decreased:",
        info["remaining_energy"] < initial_energy
    )


def main():

    print("\n" + "#" * 60)
    print("AUV ENVIRONMENT VALIDATION TEST")
    print("#" * 60)

    test_observation_and_action_space()

    test_goal_reaching()

    test_collision()

    test_energy_system()

    print("\n" + "#" * 60)
    print("VALIDATION TEST COMPLETED")
    print("#" * 60)


if __name__ == "__main__":

    main()