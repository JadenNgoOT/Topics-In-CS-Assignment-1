import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from vizdoom_env import make_vizdoom_env


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent in VizDoom")
    parser.add_argument("--scenario", type=str, default="basic",
                        help="Scenario name (e.g. basic, deadly_corridor, defend_the_center)")
    parser.add_argument("--reward", type=str, default="fast_kill",
                        help="Reward shaping type (e.g. fast_kill, survival, default)")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps")
    parser.add_argument("--render", action="store_true",
                        help="Render VizDoom window during training")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments for vectorized training")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    args = parser.parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = np.random.randint(0, 100000)
        print(f"No seed specified. Using random seed: {args.seed}")
    else:
        print(f"Using provided seed: {args.seed}")

    np.random.seed(args.seed)

    print(f"\nCreating {args.num_envs} VizDoom envs for scenario '{args.scenario}' with reward '{args.reward}'")

    # Factory function for each environment instance
    def make_env(rank):
        def _init():
            env = make_vizdoom_env(scenario=args.scenario,
                                   render=args.render if rank == 0 else False,
                                   reward_type=args.reward)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    # Create vectorized environment
    if args.num_envs > 1:
        env_fns = [make_env(i) for i in range(args.num_envs)]
        try:
            env = SubprocVecEnv(env_fns)
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")
            env = DummyVecEnv(env_fns)
    else:
        env = make_vizdoom_env(scenario=args.scenario,
                               render=args.render,
                               reward_type=args.reward)
        env.reset(seed=args.seed)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Optional environment validation
    if args.num_envs == 1:
        check_env(env, warn=True)

    # Create PPO model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"./logs_vizdoom/{args.scenario}",
        n_steps=1024,
        batch_size=64,
        learning_rate=4.25e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.02,  # encourages exploration
        seed=args.seed,  # ensures deterministic initialization
    )

    print("\nStarting training...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save model
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/ppo_vizdoom_{args.scenario}_{args.reward}.zip"
    model.save(model_path)

    print(f"\nTraining completed and model saved at: {model_path}")
    env.close()


if __name__ == "__main__":
    main()
