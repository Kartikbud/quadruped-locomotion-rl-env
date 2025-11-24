import time
import numpy as np
from env import QuadEnv   # ← replace with your actual filename

def main():
    # Create environment
    env = QuadEnv()
    print("✅ Environment initialized")

    # Reset environment
    obs, info = env.reset()
    print(f"Reset successful — observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Open MuJoCo viewer
    print("🎥 Launching MuJoCo viewer... (Close the viewer window or press Ctrl+C to quit)")

    # Run for a few seconds of simulated time
    sim_time = 100.0  # seconds
    control_freq = env.frequency
    steps = int(sim_time * control_freq)

    start_time = time.time()
    for i in range(steps):
        # Random action from the action space
        #action = env.action_space.sample()
        action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.5]
        obs, reward, terminated, truncated, info = env.step(action)

        # Print progress occasionally
        if i % int(control_freq) == 0:
            print(f"t={i/env.frequency:4.1f}s | reward={reward:+.4f} | term={terminated} | trunc={truncated}")

        # Render one frame
        env.render(mode="human")

        if terminated or truncated:
            print("Episode ended — resetting environment.")
            obs, info = env.reset()

        # Maintain roughly real-time simulation
        time.sleep(1.0 / control_freq)

    total_time = time.time() - start_time
    print(f"\n✅ Simulation completed ({steps} control steps in {total_time:.2f}s)")

    env.close()
    print("Environment closed cleanly.")

if __name__ == "__main__":
    main()
