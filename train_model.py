import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th

from quadruped.env import QuadEnv
from quadruped.training.callbacks import RandomizationCallback, RewardLoggingCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO policy for quadruped env.")
    parser.add_argument(
        "--training-time",
        type=int,
        default=5_000_000,
        help="Total environment steps to train.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel SubprocVecEnv workers.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PPO_1.zip",
        help="Filename for the final model zip (saved under saved_models/final_models).",
    )
    return parser.parse_args()


def make_env():
    return QuadEnv()


def main():
    args = parse_args()
    model_name = args.model_name if args.model_name.endswith(".zip") else f"{args.model_name}.zip"
    model_stem = Path(model_name).stem

    logs_dir = Path("logs") / model_stem
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path("saved_models/checkpoints") / model_stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([make_env for _ in range(args.num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(logs_dir),
        device="auto",
    )

    rollout_size = model.n_steps * args.num_envs
    log_every_n_steps = max(64_000, rollout_size)
    rand_callback = RandomizationCallback()
    reward_log_callback = RewardLoggingCallback(window_size=100)
    log_callback = LogEveryNTimesteps(n_steps=log_every_n_steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix=model_stem,
    )
    
    model.learn(
        total_timesteps=args.training_time,
        callback=[rand_callback, reward_log_callback, log_callback, checkpoint_callback],
        log_interval=None,
    )

    model_save_dir = Path("saved_models/final_models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / model_name
    model.save(str(model_save_path))
    env.close()


if __name__ == "__main__":
    main()
