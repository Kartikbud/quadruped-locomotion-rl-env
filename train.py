import gymnasium as gym
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env import QuadEnv

def main():

    save_checkpoints = False

    num_envs = 4
    def make_env():
        return QuadEnv()
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)]) #running parallel environments

    class RandomizationCallback(BaseCallback): #defining the callback so that the domain and dynamics are resampled every 64 000 training steps
        def __init__(self, steps_between=64_000, verbose=0):
            super().__init__(verbose)
            self.steps_between = steps_between
            self._steps_since = 0
            self._rollout_size = None

        def _on_training_start(self):
            if self.training_env is not None and self.model is not None:
                num_envs = getattr(self.training_env, "num_envs", 1)
                self._rollout_size = self.model.n_steps * num_envs
            else:
                self._rollout_size = 0
            return True

        def _resample(self):
            if self.training_env is not None:
                self.training_env.env_method("resample_randomization")

        def _on_rollout_start(self):
            if self._rollout_size is None:
                self._on_training_start()
            self._steps_since += self._rollout_size or 0
            if self._steps_since >= self.steps_between:
                self._resample()
                self._steps_since = 0
            return True

        def _on_step(self):
            return True

    # saving a model every 50k steps as a checkpoint
    if save_checkpoints:
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,  # save every 50k steps
            save_path="./checkpoints/",
            name_prefix="ppo_quad_checkpoint"
        )
        
    rand_callback = RandomizationCallback()

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
    total_timesteps = 5_000_000

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, rand_callback]
    )

    # Saving the final model with sequential PPO_<n> naming
    model_save_path = get_next_model_path()
    model.save(str(model_save_path))

    env.close()

def get_next_model_path(directory: str = "final_models", prefix: str = "PPO_") -> Path: #helper function to save models to the final_models folder with the correct name based on the models already solved
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.zip$")
    max_index = 0
    for file in folder.glob(f"{prefix}*.zip"):
        match = pattern.fullmatch(file.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    next_index = max_index + 1 if max_index > 0 else 1
    return folder / f"{prefix}{next_index}"

if __name__ == "__main__":
    main()
