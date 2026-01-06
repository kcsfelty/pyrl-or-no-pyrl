"""Common training helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory


@tf.function
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def compute_avg_return(environment, policy, num_episodes: int = 10) -> float:
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            reward = time_step.reward.numpy()
            episode_return += float(np.mean(reward))
        total_return += episode_return
    return total_return / num_episodes


def make_random_policy(env):
    return random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
