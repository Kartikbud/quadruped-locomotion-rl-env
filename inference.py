import time
from stable_baselines3 import PPO
from env import QuadEnv

def main():
    env = QuadEnv()
    model = PPO.load("final_models/PPO_2", env=env)

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
