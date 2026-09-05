from stable_baselines3 import PPO


class PPOAgent:
    """
    Creates and manages the PPO reinforcement
    learning agent for the AUV.
    """

    def __init__(self, environment):

        self.environment = environment

        self.model = PPO(
            policy="MlpPolicy",
            env=self.environment,
            verbose=1
        )

    def train(self, total_timesteps):

        self.model.learn(
            total_timesteps=total_timesteps
        )

    def save(self, path):

        self.model.save(path)

    def load(self, path):

        self.model = PPO.load(
            path,
            env=self.environment
        )

    def predict(self, observation):

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        return action