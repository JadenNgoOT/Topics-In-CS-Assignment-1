import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vizdoom_env import make_vizdoom_env

def main():
    print("Creating VizDoom environment...")
    
    # Create environment
    env = make_vizdoom_env(render=False)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test environment reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min/max: {obs.min()}/{obs.max()}")
    
    # Check if environment is valid
    print("\nChecking environment...")
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return
    
    print("\nCreating PPO model...")
    
    # Create model
    model = PPO(
        policy="CnnPolicy",  # CNN for pixel input
        env=env,
        verbose=1,
        tensorboard_log="./logs_vizdoom",
        n_steps=2048,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
    )
    
    print("Model created successfully!")
    print("Starting training...")
    
    # Train the model
    model.learn(total_timesteps=100_000, progress_bar=True)  # Reduced for testing
    
    # Save the model
    os.makedirs("./models", exist_ok=True)
    model.save("./models/ppo_vizdoom_basic")
    
    print("Training completed and model saved!")
    env.close()

if __name__ == "__main__":
    main()