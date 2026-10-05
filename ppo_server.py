import json
import socket

import numpy as np
from stable_baselines3 import PPO


HOST = "127.0.0.1"
PORT = 5005

MODEL_PATH = "models/auv_ppo_improved_safety.zip"


print("=" * 60)
print("PPO SERVER")
print("=" * 60)

print("Loading PPO model...")
model = PPO.load(MODEL_PATH, device="cpu")

print("PPO model loaded successfully.")
print("Observation space:", model.observation_space)
print("Action space:", model.action_space)
print(f"Listening on {HOST}:{PORT}")
print("=" * 60)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for Unity bridge...")

    connection, address = server.accept()

    with connection:
        print("Bridge connected:", address)

        buffer = ""

        while True:
            data = connection.recv(4096)

            if not data:
                print("Bridge disconnected.")
                break

            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)

                if not line.strip():
                    continue

                request = json.loads(line)

                observation = np.asarray(
                    request["observation"],
                    dtype=np.float32
                )

                if observation.shape != (28,):
                    raise ValueError(
                        f"Expected observation shape (28,), "
                        f"got {observation.shape}"
                    )

                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                action = np.asarray(
                    action,
                    dtype=np.float32
                ).reshape(-1)

                if action.shape != (3,):
                    raise ValueError(
                        f"Expected action shape (3,), "
                        f"got {action.shape}"
                    )

                response = {
                    "action": action.tolist()
                }

                connection.sendall(
                    (json.dumps(response) + "\n").encode("utf-8")
                )

                print(
                    "Observation received:",
                    observation.shape,
                    "→ Action:",
                    action
                )