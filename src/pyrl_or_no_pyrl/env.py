"""Deal or No Deal environment for TF-Agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .offers import banker_offer


DEFAULT_CASE_VALUES = (
    0.01,
    1,
    5,
    10,
    25,
    50,
    75,
    100,
    200,
    300,
    400,
    500,
    750,
    1000,
    5000,
    10000,
    25000,
    50000,
    75000,
    100000,
    200000,
    300000,
    400000,
    500000,
    750000,
    1000000,
)


@dataclass
class DealOrNoDealConfig:
    case_values: Sequence[float] = DEFAULT_CASE_VALUES
    max_steps: int | None = None


class DealOrNoDealEnv(py_environment.PyEnvironment):
    """Minimal Deal or No Deal environment.

    Actions: 0 = Deal, 1 = No Deal
    Observation: [remaining_values..., current_offer]
    """

    def __init__(self, config: DealOrNoDealConfig | None = None, seed: int | None = None):
        super().__init__()
        self._config = config or DealOrNoDealConfig()
        self._values = np.asarray(self._config.case_values, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._n_cases = len(self._values)
        self._max_steps = self._config.max_steps or (self._n_cases - 1)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._n_cases + 1,),
            dtype=np.float32,
            minimum=0.0,
            maximum=float(np.max(self._values)),
            name="observation",
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self._remaining_mask = np.ones(self._n_cases, dtype=np.float32)
        self._player_index = int(self._rng.integers(0, self._n_cases))
        self._step_count = 0
        self._offer = banker_offer(self._remaining_values(), self._step_count, self._max_steps)
        self._episode_ended = False

    def _remaining_values(self) -> np.ndarray:
        return self._values[self._remaining_mask.astype(bool)]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_observation(self) -> np.ndarray:
        masked = self._values * self._remaining_mask
        return np.concatenate([masked, np.array([self._offer], dtype=np.float32)])

    def _reset(self):
        self._reset_state()
        return ts.restart(self._get_observation())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action = int(np.asarray(action).reshape(-1)[0])
        if action == 0:
            self._episode_ended = True
            reward = float(self._offer)
            return ts.termination(self._get_observation(), reward)

        # No Deal: open one random non-player case.
        available = np.where(self._remaining_mask > 0.0)[0]
        available = available[available != self._player_index]
        if available.size > 0:
            opened = int(self._rng.choice(available))
            self._remaining_mask[opened] = 0.0

        self._step_count += 1
        remaining = self._remaining_values()
        if remaining.size <= 1 or self._step_count >= self._max_steps:
            self._episode_ended = True
            reward = float(self._values[self._player_index])
            return ts.termination(self._get_observation(), reward)

        self._offer = banker_offer(remaining, self._step_count, self._max_steps)
        return ts.transition(self._get_observation(), reward=0.0, discount=1.0)


def make_batched_env(batch_size: int = 32, seed: int | None = None) -> batched_py_environment.BatchedPyEnvironment:
    """Create a batched environment for vectorized training."""

    def _make_env(i: int) -> DealOrNoDealEnv:
        env_seed = None if seed is None else seed + i
        return DealOrNoDealEnv(seed=env_seed)

    envs: Iterable[DealOrNoDealEnv] = [_make_env(i) for i in range(batch_size)]
    return batched_py_environment.BatchedPyEnvironment(envs)
