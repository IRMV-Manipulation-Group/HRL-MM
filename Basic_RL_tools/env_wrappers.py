import numpy as np
import itertools
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete

from collections import deque


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


# class HistoryEnv(ProxyEnv, Env):
#     def __init__(self, wrapped_env, history_len):
#         super().__init__(wrapped_env)
#         self.history_len = history_len

#         high = np.inf * np.ones(
#             self.history_len * self.observation_space.low.size)
#         low = -high
#         self.observation_space = Box(low=low,
#                                      high=high,
#                                      )
#         self.history = deque(maxlen=self.history_len)

#     def step(self, action):
#         state, reward, done, info = super().step(action)
#         self.history.append(state)
#         flattened_history = self._get_history().flatten()
#         return flattened_history, reward, done, info

#     def reset(self, **kwargs):
#         state = super().reset()
#         self.history = deque(maxlen=self.history_len)
#         self.history.append(state)
#         flattened_history = self._get_history().flatten()
#         return flattened_history

#     def _get_history(self):
#         observations = list(self.history)

#         obs_count = len(observations)
#         for _ in range(self.history_len - obs_count):
#             dummy = np.zeros(self._wrapped_env.observation_space.low.size)
#             observations.append(dummy)
#         return np.c_[observations]


# class DiscretizeEnv(ProxyEnv, Env):
#     def __init__(self, wrapped_env, num_bins):
#         super().__init__(wrapped_env)
#         low = self.wrapped_env.action_space.low
#         high = self.wrapped_env.action_space.high
#         action_ranges = [
#             np.linspace(low[i], high[i], num_bins)
#             for i in range(len(low))
#         ]
#         self.idx_to_continuous_action = [
#             np.array(x) for x in itertools.product(*action_ranges)
#         ]
#         self.action_space = Discrete(len(self.idx_to_continuous_action))

#     def step(self, action):
#         continuous_action = self.idx_to_continuous_action[action]
#         return super().step(continuous_action)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            should_normalize=False,
            reward_scale=1.,
            ):
        ProxyEnv.__init__(self, env)
        self.should_normalize = should_normalize

        self._reward_scale = reward_scale

        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

        self.obs_low = self._wrapped_env.observation_space.low
        self.obs_high = self._wrapped_env.observation_space.high

    # def estimate_obs_stats(self, obs_batch, override_values=False):
    #     if self._obs_mean is not None and not override_values:
    #         raise Exception("Observation mean and std already set. To "
    #                         "override, set override_values to True.")
    #     self._obs_mean = np.mean(obs_batch, axis=0)
    #     self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        # scale obs to [-1, 1]
        return (obs - self.obs_low) / (self.obs_high - self.obs_low + 1e-8) * 2 - 1

    def reset(self, **kwargs):
        next_obs, info = self._wrapped_env.reset(**kwargs)
        # next_obs, info = wrapped_reset
        next_obs = np.array(next_obs)
        if self.should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, info

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)

        next_obs, reward, done, truncated, info = wrapped_step
        # print('before:{}'.format(next_obs))
        if self.should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        # print('after:{}'.format(next_obs))
        return next_obs, reward * self._reward_scale, done, truncated, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

