import argparse
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from vizdoom_env import make_vizdoom_env

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO VizDoom agent with metrics logging")
    parser.add_argument("--scenario", type=str, default="basic", help="Scenario name (e.g. basic, deadly_corridor, defend_the_center)")
    parser.add_argument("--reward", type=str, default="fast_kill", help="Reward type used during training")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", type=int, default=0, help="Render window (1=yes, 0=no)")
    parser.add_argument("--csv_out", type=str, default="logs/eval_metrics.csv", help="CSV output path")
    args = parser.parse_args()

    # Load environment
    env = make_vizdoom_env(scenario=args.scenario, render=bool(args.render), reward_type=args.reward)

    # Model path
    model_path = f"./models/ppo_vizdoom_{args.scenario}_{args.reward}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Prepare output directory
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    rows = []

    # Run evaluation episodes
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Collect metrics
        summary = env.get_episode_summary() if hasattr(env, "get_episode_summary") else {}
        row = {
            "episode": ep,
            "reward": total_reward,
            "kills": info.get("kills", 0),
            "damage_dealt": info.get("damage_dealt", 0),
            "health": info.get("health", 0),
            "time_to_kill": summary.get("time_to_kill", np.nan),
            "survival_time": summary.get("survival_time", np.nan),
        }
        rows.append(row)

        print(f"Episode {ep}: reward={row['reward']:.2f}, kills={row['kills']}, damage={row['damage_dealt']}, "
              f"time={row['survival_time']:.2f}s, ttk={row['time_to_kill']}")

    # Compute summary stats
    mean_reward = np.mean([r["reward"] for r in rows])
    mean_kills = np.mean([r["kills"] for r in rows])
    mean_damage = np.mean([r["damage_dealt"] for r in rows])

    print("\n--- Summary ---")
    print(f"Episodes: {len(rows)}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Kills: {mean_kills:.2f}")
    print(f"Mean Damage: {mean_damage:.2f}")

    # Save to CSV
    fieldnames = ["episode", "reward", "kills", "damage_dealt", "health", "time_to_kill", "survival_time"]
    csv_path = args.csv_out

    # Add scenario info automatically if user didn’t specify
    if args.csv_out == "logs/eval_metrics.csv":
        csv_path = f"logs/eval_metrics_vizdoom_{args.scenario}_{args.reward}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ Saved metrics to {csv_path}")
    env.close()

if __name__ == "__main__":
    main()
