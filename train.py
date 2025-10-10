import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
from vizdoom_env import make_vizdoom_env

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent in VizDoom")
    parser.add_argument("--scenario", type=str, default="basic", help="Scenario name (e.g. basic, deadly_corridor, defend_the_center)")
    parser.add_argument("--reward", type=str, default="fast_kill", help="Reward shaping type (e.g. fast_kill, default)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--render", action="store_true", help="Render VizDoom window during training")
    args = parser.parse_args()

    print(f"Creating VizDoom environment for scenario '{args.scenario}' with reward '{args.reward}'")

    env = make_vizdoom_env(scenario=args.scenario, render=args.render, reward_type=args.reward)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Optional environment validation
    check_env(env, warn=True)

    # Create model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"./logs_vizdoom/{args.scenario}",
        n_steps=2048,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
    )

    print("Starting training...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save model
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/ppo_vizdoom_{args.scenario}_{args.reward}.zip"
    model.save(model_path)

    print(f"Training completed and model saved at: {model_path}")
    env.close()

if __name__ == "__main__":
    main()
