import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env import QuadEnv

def main():
    # ✅ Create a headless environment (no rendering)
    env = DummyVecEnv([lambda: QuadEnv()])

    # ✅ Create an evaluation environment (optional)
    eval_env = DummyVecEnv([lambda: QuadEnv()])

    # ✅ Checkpoint callback (saves every N steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # save every 50k steps
        save_path="./checkpoints/",
        name_prefix="ppo_quad_checkpoint"
    )

    # ✅ Evaluation callback (tests model performance periodically)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=25_000,
        deterministic=True,
        render=False
    )

    # ✅ Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/",
        device="auto"
    )

    # ✅ Train without rendering
    total_timesteps = 2_500_000  # increase to 2M+ for serious training
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    # ✅ Save final model
    model.save("ppo_quad_spot")
    print("✅ Training complete — model saved to ppo_quad_spot.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
