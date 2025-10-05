import time
from stable_baselines3 import PPO
from env import QuadEnv

def main():
    # Load trained model and environment
    env = QuadEnv()
    model = PPO.load("ppo_quad_spot", env=env)

    obs, _ = env.reset()
    env.render()

    print("🎮 Starting inference loop... (Close the MuJoCo window to stop)")

    while True:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Reward: {reward:.4f}")

        if terminated or truncated:
            print("Episode ended — resetting environment.")
            obs, _ = env.reset()
            env.render()

        # Add small delay for smooth visuals
        time.sleep(1/50.0)

if __name__ == "__main__":
    main()
