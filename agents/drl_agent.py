import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers.time_limit import TimeLimit

class PPOTradingAgent:
    def __init__(
        self,
        env_class,           # Your custom env class, e.g., FuturesTradingEnv
        env_kwargs: dict,    # kwargs to init your env
        model_path=None,     # Optional path to load pretrained model
        policy="MlpPolicy",
        verbose=1,
        n_envs: int = 1,      # ← how many parallel copies of the env
        n_steps: int = 2048,  # ← rollout length per update
        batch_size: int = 64, # ← minibatch size for PPO
        learning_rate=1e-4,
        gamma=0.95,
        gae_lambda=0.92,
        clip_range=0.2,
        ent_coef=0.005,
        seed: int = None,
        tensorboard_log: str = None,
        **ppo_kwargs         # Extra PPO hyperparameters like learning_rate, n_steps etc.
    ):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.verbose = verbose
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.seed = seed

        def make_raw_env():
            env = self.env_class(**self.env_kwargs)
            # If Gym/Shimmy wrapped it in a TimeLimit, strip it off:
            if isinstance(env, TimeLimit):
                env = env.env
            return env
        
       
        # ─── VECTORIZE ──────────────────────────────────────────────────────────
        self.train_env = DummyVecEnv([make_raw_env for _ in range(self.n_envs)])


        #Seed environment

        if seed is not None:
            self.train_env.seed(seed)

        #Create or Load Model    
        if model_path:
            self.model = PPO.load(model_path, env=self.train_env)
            if self.verbose:
                print(f"Loaded PPO model from {model_path}")
        else:
            self.model = PPO(
                policy,
                env=self.train_env,
                verbose=self.verbose,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                seed=self.seed,
                tensorboard_log=tensorboard_log,  # ← new
                **ppo_kwargs
            )
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
