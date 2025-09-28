import gymnasium as gym

class FastKillRewardWrapper(gym.Wrapper):
    """
    Reward shaping for 'basic' scenario: penalize time, reward kills, optional damage bonus.
    """
    def __init__(self, env, time_penalty=-0.01, kill_bonus=1.0, damage_bonus=0.01):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.kill_bonus = kill_bonus
        self.damage_bonus = damage_bonus
        self.prev_kills = 0
        self.prev_damage = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_kills = 0
        self.prev_damage = 0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = self.time_penalty  # penalize each step

        # Reward for kill
        kills = info.get("kills", 0)
        if kills > self.prev_kills:
            shaped += self.kill_bonus * (kills - self.prev_kills)
        self.prev_kills = kills

        # Reward partial damage
        damage = info.get("damage_dealt", 0)
        if damage > self.prev_damage:
            shaped += (damage - self.prev_damage) * self.damage_bonus
        self.prev_damage = damage

        return obs, base_reward + shaped, terminated, truncated, info
