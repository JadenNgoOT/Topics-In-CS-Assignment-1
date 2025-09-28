from stable_baselines3 import PPO
import argparse, os, csv
import numpy as np
from vizdoom_env import make_vizdoom_env

env = make_vizdoom_env(scenario="basic", render=True, reward_type="fast_kill")

model = PPO.load("./models/ppo_vizdoom_basic")

for ep in range(5):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep+1}: reward={total_reward:.2f}, time={info['time']:.4f}s, kills={info['kills']}")

env.close()