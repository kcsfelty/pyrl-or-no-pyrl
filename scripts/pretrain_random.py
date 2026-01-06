"""Collect random episodes, append to buffer, and pretrain a DQN agent."""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step

from pyrl_or_no_pyrl import make_batched_env
from pyrl_or_no_pyrl.utils import configure_gpu


def collect_random_transitions(env, policy, num_episodes: int):
    obs_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    action_list: List[int] = []
    reward_list: List[float] = []
    discount_list: List[float] = []
    step_type_list: List[int] = []
    next_step_type_list: List[int] = []

    for _ in range(num_episodes):
        time_step = env.reset()
        done = False
        while not done:
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)

            obs_list.append(time_step.observation.numpy().squeeze())
            next_obs_list.append(next_time_step.observation.numpy().squeeze())
            action_list.append(int(action_step.action.numpy().squeeze()))
            reward_list.append(float(next_time_step.reward.numpy().squeeze()))
            discount_list.append(float(next_time_step.discount.numpy().squeeze()))
            step_type_list.append(int(time_step.step_type.numpy().squeeze()))
            next_step_type_list.append(int(next_time_step.step_type.numpy().squeeze()))

            done = bool(next_time_step.is_last().numpy().squeeze())
            time_step = next_time_step

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "next_obs": np.asarray(next_obs_list, dtype=np.float32),
        "action": np.asarray(action_list, dtype=np.int32),
        "reward": np.asarray(reward_list, dtype=np.float32),
        "discount": np.asarray(discount_list, dtype=np.float32),
        "step_type": np.asarray(step_type_list, dtype=np.int32),
        "next_step_type": np.asarray(next_step_type_list, dtype=np.int32),
    }


def append_buffer(path: str, new_data: dict) -> dict:
    if os.path.exists(path):
        old = np.load(path)
        merged = {
            key: np.concatenate([old[key], new_data[key]], axis=0)
            for key in new_data.keys()
        }
    else:
        merged = new_data
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **merged)
    return merged


def load_into_replay(buffer, data: dict):
    for i in range(data["obs"].shape[0]):
        time_step = ts.TimeStep(
            step_type=tf.convert_to_tensor(data["step_type"][i]),
            reward=tf.convert_to_tensor(0.0),
            discount=tf.convert_to_tensor(1.0),
            observation=tf.convert_to_tensor(data["obs"][i]),
        )
        next_time_step = ts.TimeStep(
            step_type=tf.convert_to_tensor(data["next_step_type"][i]),
            reward=tf.convert_to_tensor(data["reward"][i]),
            discount=tf.convert_to_tensor(data["discount"][i]),
            observation=tf.convert_to_tensor(data["next_obs"][i]),
        )
        action_step = policy_step.PolicyStep(
            action=tf.convert_to_tensor(data["action"][i]),
            state=(),
            info=(),
        )
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)


def pretrain_dqn(train_env, data: dict, train_steps: int):
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(128, 64),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE
        ),
        gamma=1.0,
        train_step_counter=tf.Variable(0),
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=max(10000, data["obs"].shape[0] + 1),
    )

    load_into_replay(replay_buffer, data)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    for _ in range(train_steps):
        experience, _ = next(iterator)
        agent.train(experience)

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--buffer-path", default="data/random_buffer.npz")
    parser.add_argument("--pretrain-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    configure_gpu()

    env_py = make_batched_env(batch_size=1, seed=args.seed)
    env = tf_py_environment.TFPyEnvironment(env_py)
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

    new_data = collect_random_transitions(env, policy, num_episodes=args.episodes)
    data = append_buffer(args.buffer_path, new_data)

    train_env_py = make_batched_env(batch_size=args.batch_size, seed=args.seed)
    train_env = tf_py_environment.TFPyEnvironment(train_env_py)

    agent = pretrain_dqn(train_env, data, train_steps=args.pretrain_steps)

    print("Random buffer size:", data["obs"].shape[0])
    print("Pretrain steps:", args.pretrain_steps)
    print("Agent train_step:", int(agent.train_step_counter.numpy()))


if __name__ == "__main__":
    main()
