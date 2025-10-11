import argparse
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from vizdoom_env import make_vizdoom_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO VizDoom agent with unified metrics logging")
    parser.add_argument("--scenario", type=str, default="basic",
                        help="Scenario name (e.g. basic, deadly_corridor, defend_the_center)")
    parser.add_argument("--reward", type=str, default="fast_kill",
                        help="Reward type used during training (e.g. fast_kill, survival)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", type=int, default=0,
                        help="Render window (1=yes, 0=no)")
    parser.add_argument("--csv_out", type=str, default="logs/eval_metrics.csv",
                        help="CSV output path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible evaluation (default: random)")
    args = parser.parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = np.random.randint(0, 100000)
        print(f"No seed specified. Using random seed: {args.seed}")
    else:
        print(f"Using provided seed: {args.seed}")
    np.random.seed(args.seed)

    # Load environment
    env = make_vizdoom_env(scenario=args.scenario, render=bool(args.render), reward_type=args.reward)
    env.reset(seed=args.seed)

    # Load trained model
    model_path = f"./models/ppo_vizdoom_{args.scenario}_{args.reward}_seed{args.seed}.zip"
    if not os.path.exists(model_path):
        # fallback if model without seed suffix
        model_path = f"./models/ppo_vizdoom_{args.scenario}_{args.reward}.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model from {model_path}")
    model = PPO.load(model_path)

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    rows = []

    print(f"\nEvaluating {args.episodes} episode(s) on scenario '{args.scenario}' (seed={args.seed})...\n")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        summary = env.get_episode_summary() if hasattr(env, "get_episode_summary") else {}

        row = {
            "episode": ep,
            "reward": total_reward,
            "kills": info.get("kills", 0),
            "health": info.get("health", 0),
            "survival_time": summary.get("time_alive", np.nan), 
            "completion_time": summary.get("completion_time", np.nan),
        }

        rows.append(row)

        print(f"Episode {ep}: reward={row['reward']:.2f}, kills={row['kills']}, "
            f"health={row['health']}, survival={row['survival_time']:.2f}s, "
            f"complete={row['completion_time']:.2f}s")

    # Compute averages
    mean_reward = np.mean([r["reward"] for r in rows])
    mean_kills = np.mean([r["kills"] for r in rows])
    mean_health = np.mean([r["health"] for r in rows])
    mean_survival = np.mean([r["survival_time"] for r in rows])

    print("\nEvaluation Summary")
    print(f"Episodes: {len(rows)}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Kills: {mean_kills:.2f}")
    print(f"Mean Health: {mean_health:.2f}")
    print(f"Mean Survival Time: {mean_survival:.2f}s")

    fieldnames = ["episode", "reward", "kills", "health", "survival_time", "completion_time"]
    csv_path = args.csv_out

    if args.csv_out == "logs/eval_metrics.csv":
        csv_path = f"logs/eval_metrics_vizdoom_{args.scenario}_{args.reward}_seed{args.seed}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
