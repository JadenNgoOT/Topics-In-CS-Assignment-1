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
    
class SurvivalRewardWrapper(gym.Wrapper):
    """
    Reward shaping for survival-type scenarios (e.g., take_cover, health_gathering).
    Rewards staying alive longer, penalizes taking damage or dying early.
    """
    def __init__(self, env, alive_bonus=0.05, damage_penalty=-0.02, death_penalty=-1.0, health_bonus=0.02):
        super().__init__(env)
        self.alive_bonus = alive_bonus
        self.damage_penalty = damage_penalty
        self.death_penalty = death_penalty
        self.health_bonus = health_bonus
        self.prev_health = None
        self.prev_damage = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_health = info.get("health", 100)  # default 100 if not provided
        self.prev_damage = 0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0

        # Reward for surviving each step
        shaped += self.alive_bonus

        # Penalize additional damage taken
        damage = info.get("damage_dealt", 0)
        if damage > self.prev_damage:
            shaped += (self.prev_damage - damage) * self.damage_penalty  # negative
        self.prev_damage = damage

        # Reward gaining health (if possible)
        health = info.get("health", self.prev_health)
        if health > self.prev_health:
            shaped += (health - self.prev_health) * self.health_bonus
        elif health < self.prev_health:
            shaped += (health - self.prev_health) * self.damage_penalty
        self.prev_health = health

        # Death penalty if episode ended due to termination
        if terminated:
            shaped += self.death_penalty

        return obs, base_reward + shaped, terminated, truncated, info
