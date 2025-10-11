import gymnasium as gym
import numpy as np

import gymnasium as gym

class FastKillRewardWrapper(gym.Wrapper):
    """
    Reward shaping for aggressive, kill-focused behavior.
    Encourages quick kills, consistent enemy engagement, and efficient combat.
    Suitable for scenarios like 'basic', 'deadly_corridor', or 'defend_the_center'.
    """

    def __init__(
        self,
        env,
        time_penalty=-0.002,   # Small penalty each step to discourage idling
        kill_bonus=1.0,        # Reward for each confirmed kill
        damage_bonus=0.02,     # Reward per unit of damage dealt
        visible_bonus=0.02     # Bonus for having enemies visible on screen
    ):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.kill_bonus = kill_bonus
        self.damage_bonus = damage_bonus
        self.visible_bonus = visible_bonus

        # Track progress across steps
        self.prev_kills = 0
        self.prev_damage = 0

    def reset(self, **kwargs):
        """
        Reset episode-specific tracking variables.
        """
        obs, info = self.env.reset(**kwargs)
        self.prev_kills = 0
        self.prev_damage = 0
        return obs, info

    def step(self, action):
        """
        Apply the agent's action, compute base environment reward,
        and add custom shaping terms that promote offensive behavior.
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0

        # Time penalty — discourage wasting time / standing idle
        shaped += self.time_penalty

        # Visibility bonus — reward if at least one enemy is visible on screen
        visible_bonus = 0.0
        state = self.env.game.get_state()
        if state and state.objects:
            if any(obj.visible for obj in state.objects):
                visible_bonus = self.visible_bonus
        shaped += visible_bonus

        # Kill reward — major reward when kill count increases
        kills = info.get("kills", 0)
        if kills > self.prev_kills:
            shaped += self.kill_bonus * (kills - self.prev_kills)
        self.prev_kills = kills

        # Damage reward — incremental reward for each unit of damage dealt
        damage = info.get("damage_dealt", 0)
        if damage > self.prev_damage:
            shaped += (damage - self.prev_damage) * self.damage_bonus
        self.prev_damage = damage

        # Combine all with base game reward (usually small or 0 in VizDoom)
        return obs, base_reward + shaped, terminated, truncated, info

    
class SurvivalRewardWrapper(gym.Wrapper):
    """
    Reward shaping for survival-oriented scenarios (e.g., defend_the_line, take_cover, health_gathering).
    Encourages movement, dodging damage, and staying alive longer.
    """

    def __init__(
        self,
        env,
        alive_bonus=0.05,        # reward per step alive
        move_bonus=0.03,         # reward for movement
        damage_penalty=-0.3,     # penalty per damage taken
        death_penalty=-2.0,      # large penalty on death
        health_bonus=0.1,        # reward for regaining health
        idle_penalty=-0.05,      # penalty for standing still too long
        idle_steps_threshold=8   # number of steps before idle penalty applies
    ):
        super().__init__(env)
        self.alive_bonus = alive_bonus
        self.move_bonus = move_bonus
        self.damage_penalty = damage_penalty
        self.death_penalty = death_penalty
        self.health_bonus = health_bonus
        self.idle_penalty = idle_penalty
        self.idle_steps_threshold = idle_steps_threshold

        # internal state
        self.prev_health = None
        self.prev_pos = None
        self.idle_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_health = info.get("health", 100)
        self.prev_pos = self._get_position()
        self.idle_steps = 0
        return obs, info

    def _get_position(self):
        """
        Extract approximate player position if the scenario exposes coordinates.
        Many VizDoom scenarios use the first 3 game variables as (posX, posY, posZ).
        """
        if hasattr(self.env, "game"):
            state = self.env.game.get_state()
            if state and len(state.game_variables) >= 2:
                return np.array(state.game_variables[:2])
        return None

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0

        # 1Survival incentive — reward just for staying alive
        shaped += self.alive_bonus

        # Encourage movement (if positional info exists)
        current_pos = self._get_position()
        if current_pos is not None and self.prev_pos is not None:
            dist = np.linalg.norm(current_pos - self.prev_pos)
            if dist > 1e-3:  # Moved enough to count
                shaped += self.move_bonus
                self.idle_steps = 0
            else:
                self.idle_steps += 1
                if self.idle_steps >= self.idle_steps_threshold:
                    shaped += self.idle_penalty
        self.prev_pos = current_pos

        # Health management — penalize taking damage, reward regaining health
        health = info.get("health", self.prev_health)
        if health > self.prev_health:
            shaped += (health - self.prev_health) * self.health_bonus
        elif health < self.prev_health:
            shaped += (health - self.prev_health) * self.damage_penalty
        self.prev_health = health

        # Death penalty
        if terminated:
            shaped += self.death_penalty

        # Combine base and shaped reward
        return obs, base_reward + shaped, terminated, truncated, info

