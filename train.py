import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import QuadEnv

def main():
    num_envs = 4
    def make_env():
        return QuadEnv()
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])

    # saving a model every 50k steps as a checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # save every 50k steps
        save_path="./checkpoints/",
        name_prefix="ppo_quad_checkpoint"
    )

    # intiialize model from SB3
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

    # training without rendering
    total_timesteps = 7_500_000  # increase to 2M+ for serious training
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback]
    )

    # Saving the final model
    model.save("ppo_quad_spot")

    env.close()

if __name__ == "__main__":
    main()
