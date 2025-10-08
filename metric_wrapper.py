# metric_wrapper.py
import gymnasium as gym
import time
import numpy as np
import vizdoom as vzd

class MetricCollectorWrapper(gym.Wrapper):
    """
    Collects rich gameplay metrics for ViZDoom episodes.
    Tracks real elapsed time, simulated time, time-to-kill, survival time,
    and aggregates damage, kills, etc.
    """

    def __init__(self, env):
        super().__init__(env)
        self.start_time = None
        self.last_time = 0.0
        self.kill_time = None
        self.prev_kills = 0
        self.prev_damage = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.last_time = 0.0
        self.kill_time = None
        self.prev_kills = 0
        self.prev_damage = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update times
        real_elapsed = time.time() - self.start_time
        sim_time = self.env.game.get_episode_time() / 35.0 if hasattr(self.env, "game") else np.nan

        # Get variables (safe fallback if missing)
        kills = info.get("kills", 0)
        damage = info.get("damage_dealt", 0)
        health = info.get("health", 0)
        ammo = info.get("ammo", 0)

        # Detect first kill and record kill_time
        if self.kill_time is None and kills > self.prev_kills:
            self.kill_time = real_elapsed

        # Update stored values
        self.prev_kills = kills
        self.prev_damage = damage
        self.last_time = real_elapsed

        # Augment info with metrics
        info.update({
            "real_time": real_elapsed,
            "sim_time": sim_time,
            "time_to_kill": self.kill_time,
            "survival_time": real_elapsed,
            "kills": kills,
            "damage_dealt": damage,
            "health": health,
            "ammo": ammo,
        })

        return obs, reward, terminated, truncated, info

    def get_episode_summary(self):
        """Return a summary dict for CSV logging (after an episode ends)."""
        return {
            "time_to_kill": self.kill_time,
            "survival_time": self.last_time,
            "total_kills": self.prev_kills,
            "total_damage": self.prev_damage,
        }
