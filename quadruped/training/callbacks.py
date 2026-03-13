from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np


class RandomizationCallback(BaseCallback):
    def __init__(self, steps_between: int = 64_000, verbose: int = 0):
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


class RewardLoggingCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self._episode_returns = None
        self._episode_lengths = None
        self._return_window = deque(maxlen=window_size)
        self._length_window = deque(maxlen=window_size)
        self._episodes_seen = 0

    def _on_training_start(self) -> None:
        num_envs = getattr(self.training_env, "num_envs", 1)
        self._episode_returns = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_)
        if rewards.size == 0 or dones.size == 0:
            return True

        self._episode_returns += rewards
        self._episode_lengths += 1

        done_idxs = np.where(dones)[0]
        for idx in done_idxs:
            ep_ret = float(self._episode_returns[idx])
            ep_len = int(self._episode_lengths[idx])
            self._return_window.append(ep_ret)
            self._length_window.append(ep_len)
            self._episodes_seen += 1
            self._episode_returns[idx] = 0.0
            self._episode_lengths[idx] = 0

        if self._return_window:
            self.logger.record("custom/ep_rew_mean", float(np.mean(self._return_window)))
            self.logger.record("custom/ep_len_mean", float(np.mean(self._length_window)))
            self.logger.record("custom/episodes_seen", self._episodes_seen)
        return True
