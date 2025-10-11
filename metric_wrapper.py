import gymnasium as gym
import time
import numpy as np

class MetricCollectorWrapper(gym.Wrapper):
    """
    Collects universal metrics across all VizDoom scenarios:
      - time_alive
      - kills
      - total_reward
      - completion_time (if scenario is completable)
      - health_remaining
    """

    def __init__(self, env):
        super().__init__(env)
        self.start_time = None
        self.total_reward = 0.0
        self.prev_kills = 0
        self.prev_health = 0
        self.last_time = 0.0
        self.completed = False
        self.completion_time = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.total_reward = 0.0
        self.prev_kills = 0
        self.prev_health = info.get("health", 100)
        self.last_time = 0.0
        self.completed = False
        self.completion_time = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track elapsed time
        real_elapsed = time.time() - self.start_time
        self.total_reward += reward

        # Collect info safely
        kills = info.get("kills", 0)
        health = info.get("health", self.prev_health)

        # Detect level completion
        if terminated and not truncated:
            self.completed = True
            self.completion_time = real_elapsed

        # Update stored values
        self.prev_kills = kills
        self.prev_health = health
        self.last_time = real_elapsed

        # Update info dictionary
        info.update({
            "time_alive": real_elapsed,
            "kills": kills,
            "total_reward": self.total_reward,
            "completion_time": self.completion_time,
            "health_remaining": health,
        })

        return obs, reward, terminated, truncated, info

    def get_episode_summary(self):
        """Summary for CSV logging or evaluation"""
        return {
            "time_alive": self.last_time,
            "kills": self.prev_kills,
            "total_reward": self.total_reward,
            "completion_time": self.completion_time,
            "health_remaining": self.prev_health,
        }
