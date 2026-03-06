from stable_baselines3.common.callbacks import BaseCallback


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
