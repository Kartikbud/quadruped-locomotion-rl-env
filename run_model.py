import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from quadruped.env.env import QuadEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained PPO model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="PPO_1.zip",
        help="Model filename from saved_models/final_models.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name if args.model_name.endswith(".zip") else f"{args.model_name}.zip"
    model_path = Path("saved_models/final_models") / model_name

    env = QuadEnv()
    model = PPO.load(str(model_path), env=env)

    obs, _ = env.reset()
    env.render()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Reward: {reward:.4f}")

        if terminated or truncated:
            print("Episode ended — resetting environment.")
            obs, _ = env.reset()
            env.render()

        # small delay for smooth visuals
        time.sleep(1/50.0)

if __name__ == "__main__":
    main()
