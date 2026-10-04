import numpy as np

print("1. NumPy:", np.__version__, flush=True)

from stable_baselines3 import PPO

print("2. PPO imported", flush=True)

# Compatibility alias for models saved with newer NumPy
import sys
sys.modules["numpy._core"] = np.core

print("3. NumPy compatibility alias set", flush=True)

MODEL_PATH = r".\models\auv_ppo_improved_safety.zip"

print("4. Loading model...", flush=True)

model = PPO.load(MODEL_PATH)

print("5. MODEL LOADED SUCCESSFULLY", flush=True)
print("OBS:", model.observation_space, flush=True)
print("ACT:", model.action_space, flush=True)