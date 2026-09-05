import numpy as np

from environment.auv_env import AUVEnvironment
from simulation.visualization import AUVVisualizer


# -------------------------------------------------
# Create and reset environment
# -------------------------------------------------

env = AUVEnvironment()
observation, info = env.reset()

# Store AUV path
path = [env.auv_position.copy()]

# Track final episode status
episode_result = "Unknown"


# -------------------------------------------------
# Simple goal-directed navigation for baseline testing
# -------------------------------------------------

for _ in range(env.max_steps):

    # Calculate direction towards the goal
    direction = env.goal_position - env.auv_position

    # Calculate distance to goal
    distance = np.linalg.norm(direction)

    # Stop if already close enough to the goal
    if distance <= env.goal_radius:
        episode_result = "Goal reached"
        break

    # Normalize direction
    action = direction / distance

    # Perform one environment step
    observation, reward, terminated, truncated, info = env.step(
        action.astype(np.float32)
    )

    # Store the new AUV position
    path.append(env.auv_position.copy())

    # Check episode result
    if info["reached_goal"]:
        episode_result = "Goal reached"
        break

    if info["collision"]:
        episode_result = "Collision with obstacle"
        break

    if truncated:
        episode_result = "Maximum steps reached"
        break


# -------------------------------------------------
# Print episode results
# -------------------------------------------------

print("\n" + "=" * 50)
print("AUV NAVIGATION EPISODE RESULT")
print("=" * 50)

print("Result:", episode_result)
print("Final AUV position:", env.auv_position)
print("Goal position:", env.goal_position)
print("Steps taken:", env.current_step)
print("Final reward:", reward)

print("=" * 50)


# -------------------------------------------------
# Visualize the environment
# -------------------------------------------------

visualizer = AUVVisualizer(env)

visualizer.plot_environment(path)