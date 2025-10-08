import gymnasium as gym

class FastKillRewardWrapper(gym.Wrapper):
    def __init__(self, env, time_penalty=-0.002, kill_bonus=1.0, damage_bonus=0.02, visible_bonus=0.02):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.kill_bonus = kill_bonus
        self.damage_bonus = damage_bonus
        self.visible_bonus = visible_bonus
        self.prev_kills = 0
        self.prev_damage = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_kills = 0
        self.prev_damage = 0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = self.time_penalty  # penalize time
        
        # Check visibility
        visible_bonus = 0.0
        state = self.env.game.get_state()
        if state and state.objects:
            if any(obj.visible for obj in state.objects):
                visible_bonus = self.visible_bonus
        shaped += visible_bonus

        # Reward for kills
        kills = info.get("kills", 0)
        if kills > self.prev_kills:
            shaped += self.kill_bonus
        self.prev_kills = kills

        # Reward for incremental damage
        damage = info.get("damage_dealt", 0)
        if damage > self.prev_damage:
            shaped += (damage - self.prev_damage) * self.damage_bonus
        self.prev_damage = damage

        return obs, base_reward + shaped, terminated, truncated, info
