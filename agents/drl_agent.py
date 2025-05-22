import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PPOTradingAgent:
    def __init__(
        self,
        env_class,           # Your custom env class, e.g., FuturesTradingEnv
        env_kwargs: dict,    # kwargs to init your env
        model_path=None,     # Optional path to load pretrained model
        policy="MlpPolicy",
        verbose=1,
        **ppo_kwargs         # Extra PPO hyperparameters like learning_rate, n_steps etc.
    ):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.verbose = verbose
        
        # Create vectorized env for training
        self.train_env = DummyVecEnv([lambda: self.env_class(**self.env_kwargs)])
        
        if model_path:
            self.model = PPO.load(model_path, env=self.train_env)
            if self.verbose:
                print(f"Loaded PPO model from {model_path}")
        else:
            self.model = PPO(policy, env=self.train_env, verbose=verbose, **ppo_kwargs)
            if self.verbose:
                print("Initialized new PPO model.")

    def train(self, total_timesteps=100_000, callback=None):
        if self.verbose:
            print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps,callback=callback)
        if self.verbose:
            print("Training completed.")

    def save(self, path):
        self.model.save(path)
        if self.verbose:
            print(f"Model saved to {path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.train_env)
        if self.verbose:
            print(f"Model loaded from {path}")

    def predict(self, env=None, deterministic=True):
        """
        Run prediction on the given environment.
        If no env is provided, uses self.env_class with self.env_kwargs.
        Returns list of actions, rewards, and done flags.
        """
        if env is None:
            env = self.env_class(**self.env_kwargs)
        obs = env.reset()
        
        actions = []
        rewards = []
        dones = []
        
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        return actions, rewards, dones

    def evaluate(self, env=None, deterministic=True):
        """
        Run prediction and summarize performance.
        Returns total reward accumulated.
        """
        actions, rewards, dones = self.predict(env, deterministic)
        total_reward = sum(rewards)
        if self.verbose:
            print(f"Evaluation finished. Total reward: {total_reward}")
        return total_reward
