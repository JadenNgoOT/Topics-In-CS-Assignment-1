import gymnasium as gym
import vizdoom as vzd
import numpy as np
from gymnasium import spaces
from collections import deque
from reward_wrappers import FastKillRewardWrapper
from metric_wrapper import MetricCollectorWrapper
from itertools import product

class VizDoomEnv(gym.Env):
    """Custom VizDoom environment wrapper for Gymnasium with built-in preprocessing"""
    
    def __init__(self, scenario="basic", render=False, frame_stack=4, frame_size=84):
        super(VizDoomEnv, self).__init__()
        
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        
        # Initialize VizDoom game
        self.game = vzd.DoomGame()
        
        # Set scenario configuration 
        self.game.load_config(f"{scenario}.cfg")
        
        # Set render mode
        if render:
            self.game.set_window_visible(True)
            self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
        else:
            self.game.set_window_visible(False)
            self.game.set_mode(vzd.Mode.PLAYER)
        
        # Initialize the game
        self.game.init()

        #test line        
        print("Available buttons:", [str(b) for b in self.game.get_available_buttons()])

        
        # Get available actions
        self.actions = []
        # Create all combinations of available buttons (0/1 for each)
        button_count = self.game.get_available_buttons_size()
        # Limit action explosion — only allow up to 2 simultaneous actions
        self.actions = [
            list(action) for action in product([0, 1], repeat=button_count)
            if sum(action) <= 2 and sum(action) > 0
        ]

        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation space: stacked grayscale frames (frame_stack, height, width)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, frame_size, frame_size),
            dtype=np.uint8
        )
    
    def _preprocess_frame(self, frame):
        """Convert frame to grayscale and resize"""
        # Convert from channels-first to channels-last if needed
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Convert RGB to grayscale
        if frame.shape[2] == 3:
            gray = np.dot(frame[..., :3], [0.2989, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = frame[:, :, 0].astype(np.uint8)
        
        # Resize using nearest neighbor
        h, w = gray.shape
        if h != self.frame_size or w != self.frame_size:
            row_indices = np.round(np.linspace(0, h-1, self.frame_size)).astype(int)
            col_indices = np.round(np.linspace(0, w-1, self.frame_size)).astype(int)
            gray = gray[np.ix_(row_indices, col_indices)]
        
        return gray
    
    def _get_observation(self):
        """Get current stacked frames observation"""
        return np.stack(list(self.frames), axis=0).astype(np.uint8)
    
    def step(self, action):
        # Run the action
        reward = self.game.make_action(self.actions[action], 4)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._preprocess_frame(state.screen_buffer)
        else:
            frame = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        self.frames.append(frame)
        obs = self._get_observation()

        # Build info dict
        info = {
            "kills": self.game.get_game_variable(vzd.GameVariable.KILLCOUNT),
            "damage_dealt": self.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT),
            "health": self.game.get_game_variable(vzd.GameVariable.HEALTH),
            "ammo": self.game.get_game_variable(vzd.GameVariable.AMMO2),
        }

        return obs, reward, done, False, info



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game.new_episode()
        state = self.game.get_state()
        frame = self._preprocess_frame(state.screen_buffer)
        
        # Initialize frame stack with the same frame repeated
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        
        obs = self._get_observation()
        return obs, {}
    
    def close(self):
        self.game.close()

def make_vizdoom_env(scenario="basic", render=False, reward_type="default"):
    env = VizDoomEnv(scenario=scenario, render=render, frame_stack=4, frame_size=64)

    # Plug in reward shaping dynamically
    if reward_type == "fast_kill":
        from reward_wrappers import FastKillRewardWrapper
        env = FastKillRewardWrapper(env)
    elif reward_type == "survival":
        from reward_wrappers import SurvivalRewardWrapper
        env = SurvivalRewardWrapper(env)

    # Wrap with metric collector last (so it sees all rewards, time, etc.)
    from metric_wrapper import MetricCollectorWrapper
    env = MetricCollectorWrapper(env)
    return env

# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = make_vizdoom_env(render=False)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test the environment
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min/max: {obs.min()}/{obs.max()}")
    
    # Run a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward}, Done={done}, Obs shape={obs.shape}")
        
        if done:
            obs, info = env.reset()
            print("Episode reset")
    
    env.close()
    print("Environment test completed successfully!")