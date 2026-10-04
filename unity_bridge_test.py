import numpy as np

from stable_baselines3 import PPO

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


print("=" * 60)
print("UNITY ↔ PPO BRIDGE TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load Team A PPO model
# ---------------------------------------------------------

MODEL_PATH = "models/auv_ppo_improved_safety.zip"

print("\nLoading Team A PPO model...")
model = PPO.load(MODEL_PATH)

print("PPO model loaded successfully.")

print("Observation space:", model.observation_space)
print("Action space:", model.action_space)


# ---------------------------------------------------------
# 2. Connect to Unity
# ---------------------------------------------------------

print("\nConnecting to Unity Editor...")

env = UnityEnvironment(
    file_name=None,
    seed=42,
    side_channels=[]
)

env.reset()

print("Connected to Unity successfully.")


# ---------------------------------------------------------
# 3. Verify Unity behavior
# ---------------------------------------------------------

for behavior_name, spec in env.behavior_specs.items():

    print("\nBehavior specifications:")
    print("Behavior Name:", behavior_name)

    print(
        "Observation shapes:",
        [obs.shape for obs in spec.observation_specs]
    )

    print(
        "Continuous actions:",
        spec.action_spec.continuous_size
    )

    print(
        "Discrete actions:",
        spec.action_spec.discrete_size
    )


# ---------------------------------------------------------
# 4. Read Unity observation
# ---------------------------------------------------------

print("\nReading current Unity step...")

for behavior_name, spec in env.behavior_specs.items():

    decision_steps, terminal_steps = env.get_steps(
        behavior_name
    )

    print("\nBehavior:", behavior_name)
    print(
        "Agents waiting for decision:",
        len(decision_steps)
    )

    print(
        "Agents terminated:",
        len(terminal_steps)
    )

    if len(decision_steps) == 0:
        print("No agent is waiting for a decision.")
        continue

    # Unity gives us one observation array:
    # shape = (number_of_agents, 28)

    observations = decision_steps.obs[0]

    print(
        "Unity observation shape:",
        observations.shape
    )

    # There is one AUV agent.
    observation = observations[0]

    print(
        "PPO input shape:",
        observation.shape
    )


    # -----------------------------------------------------
    # 5. Ask Team A PPO for an action
    # -----------------------------------------------------

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = np.asarray(
        action,
        dtype=np.float32
    )

    print("\nPPO action produced:")
    print(action)

    print(
        "PPO action shape:",
        action.shape
    )


    # -----------------------------------------------------
    # 6. Send PPO action to Unity
    # -----------------------------------------------------

    unity_action = action.reshape(
        1,
        spec.action_spec.continuous_size
    )

    print(
        "\nSending PPO action to Unity:",
        unity_action
    )

    env.set_actions(
        behavior_name,
        ActionTuple(
            continuous=unity_action
        )
    )

    print("PPO action sent successfully.")


# ---------------------------------------------------------
# 7. Advance Unity one step
# ---------------------------------------------------------

env.step()

print("\nUnity step completed successfully.")


# ---------------------------------------------------------
# 8. Close connection
# ---------------------------------------------------------

env.close()

print("\n" + "=" * 60)
print("UNITY ↔ PPO BRIDGE TEST PASSED")
print("=" * 60)