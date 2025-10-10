import gymnasium as gym
import numpy as np

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
    Encourages movement, dodging attacks, and staying alive longer.
    """
    def __init__(
        self,
        env,
        alive_bonus=0.02,
        move_bonus=0.01,
        damage_penalty=-0.1,
        death_penalty=-1.0,
        health_bonus=0.05,
        idle_penalty=-0.01
    ):
        super().__init__(env)
        self.alive_bonus = alive_bonus
        self.move_bonus = move_bonus
        self.damage_penalty = damage_penalty
        self.death_penalty = death_penalty
        self.health_bonus = health_bonus
        self.idle_penalty = idle_penalty

        self.prev_health = None
        self.prev_damage = 0
        self.prev_pos = None
        self.idle_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_health = info.get("health", 100)
        self.prev_damage = 0
        self.prev_pos = self._get_position()
        self.idle_steps = 0
        return obs, info

    def _get_position(self):
        """Estimate player position from VizDoom state variables."""
        if hasattr(self.env, "game"):
            state = self.env.game.get_state()
            if state:
                vars = state.game_variables
                if len(vars) >= 2:
                    return np.array(vars[:2])
        return None

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = self.alive_bonus  # reward surviving each step

        # Position change → movement reward
        current_pos = self._get_position()
        if current_pos is not None and self.prev_pos is not None:
            dist = np.linalg.norm(current_pos - self.prev_pos)
            if dist > 1e-3:  # moved enough to count
                shaped += self.move_bonus
                self.idle_steps = 0
            else:
                self.idle_steps += 1
                if self.idle_steps > 5:  # discourage long idling
                    shaped += self.idle_penalty
        self.prev_pos = current_pos

        # Health tracking
        health = info.get("health", self.prev_health)
        if health > self.prev_health:
            shaped += (health - self.prev_health) * self.health_bonus
        elif health < self.prev_health:
            shaped += (health - self.prev_health) * self.damage_penalty
        self.prev_health = health

        # Death penalty
        if terminated:
            shaped += self.death_penalty

        return obs, base_reward + shaped, terminated, truncated, info

