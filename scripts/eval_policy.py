"""Train and summarize learned policies for Deal or No Deal."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network, q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from pyrl_or_no_pyrl import make_batched_env
from pyrl_or_no_pyrl.utils import configure_gpu


@dataclass
class SummaryStats:
    avg_return: float
    median_return: float
    std_return: float
    deal_rate_per_decision: float
    deal_rate_per_episode: float
    total_episodes: int


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def build_agent(agent_name: str, train_env):
    if agent_name == "dqn":
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
        return agent

    if agent_name == "c51":
        q_net = categorical_q_network.CategoricalQNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=(128, 64),
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        agent = categorical_dqn_agent.CategoricalDqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            categorical_q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=tf.keras.losses.Huber(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            gamma=1.0,
            train_step_counter=tf.Variable(0),
        )
        return agent

    raise ValueError(f"Unsupported agent: {agent_name}")


def train_agent(agent, train_env, train_steps: int, warmup_steps: int = 200):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=50000,
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    for _ in range(warmup_steps):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    for _ in range(train_steps):
        collect_step(train_env, agent.collect_policy, replay_buffer)
        experience, _ = next(iterator)
        agent.train(experience)


def extract_features(time_step) -> Tuple[np.ndarray, float, float, float, int]:
    obs = time_step.observation.numpy().squeeze()
    offer = float(obs[-1])
    remaining_values = obs[:-1]
    remaining_values = remaining_values[remaining_values > 0]
    ev = float(np.mean(remaining_values)) if remaining_values.size else 0.0
    max_val = float(np.max(remaining_values)) if remaining_values.size else 0.0
    remaining = int(remaining_values.size)
    return remaining_values, offer, ev, max_val, remaining


def evaluate_policy(policy, eval_env, num_episodes: int):
    rows: List[Dict[str, float]] = []
    returns: List[float] = []
    deal_decisions = 0
    deal_episodes = 0

    for episode in range(num_episodes):
        time_step = eval_env.reset()
        episode_return = 0.0
        step = 0
        done = False

        took_deal = False
        while not done:
            remaining_values, offer, ev, max_val, remaining = extract_features(
                time_step
            )
            ratio = offer / ev if ev > 0 else 0.0

            action_step = policy.action(time_step)
            action = int(action_step.action.numpy().squeeze())
            next_time_step = eval_env.step(action_step.action)
            reward = float(next_time_step.reward.numpy().squeeze())
            done = bool(next_time_step.is_last().numpy().squeeze())

            if action == 0:
                deal_decisions += 1
                took_deal = True

            episode_return += reward

            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "remaining": remaining,
                    "offer": offer,
                    "ev": ev,
                    "max": max_val,
                    "ratio": ratio,
                    "action": action,
                    "reward": reward,
                    "done": int(done),
                }
            )

            time_step = next_time_step
            step += 1

        returns.append(episode_return)
        if took_deal:
            deal_episodes += 1

    stats = SummaryStats(
        avg_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        std_return=float(np.std(returns)),
        deal_rate_per_decision=float(deal_decisions / max(len(rows), 1)),
        deal_rate_per_episode=float(deal_episodes / max(len(returns), 1)),
        total_episodes=len(returns),
    )
    return rows, stats


def summarize_by_ratio(rows, bins):
    buckets: Dict[str, List[int]] = {}
    for row in rows:
        ratio = row["ratio"]
        action = row["action"]
        for i in range(len(bins) - 1):
            if bins[i] <= ratio < bins[i + 1]:
                key = f"[{bins[i]:.2f},{bins[i+1]:.2f})"
                buckets.setdefault(key, []).append(action)
                break
    summary = []
    for key, actions in buckets.items():
        deal_rate = np.mean([1 if a == 0 else 0 for a in actions])
        summary.append((key, deal_rate, len(actions)))
    return summary


def write_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "c51", "random"], default="dqn")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out-csv", default="policy_eval.csv")
    args = parser.parse_args()

    configure_gpu()

    train_py_env = make_batched_env(batch_size=args.batch_size, seed=args.seed)
    eval_py_env = make_batched_env(batch_size=1, seed=args.seed + 999)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    if args.agent == "random":
        policy = random_tf_policy.RandomTFPolicy(
            eval_env.time_step_spec(), eval_env.action_spec()
        )
    else:
        agent = build_agent(args.agent, train_env)
        agent.initialize()
        train_agent(agent, train_env, train_steps=args.train_steps)
        policy = agent.policy

    rows, stats = evaluate_policy(policy, eval_env, args.eval_episodes)
    write_csv(args.out_csv, rows)

    print("Summary")
    print(f"Episodes: {stats.total_episodes}")
    print(f"Average return: {stats.avg_return:.2f}")
    print(f"Median return: {stats.median_return:.2f}")
    print(f"Std return: {stats.std_return:.2f}")
    print(f"Deal action rate (per decision): {stats.deal_rate_per_decision:.3f}")
    print(f"Deal action rate (per episode): {stats.deal_rate_per_episode:.3f}")

    bins = [0.0, 0.6, 0.8, 1.0, 1.2, 2.0, 10.0]
    ratio_summary = summarize_by_ratio(rows, bins)
    print("\nDeal rate by offer/EV ratio")
    for key, deal_rate, count in ratio_summary:
        print(f"{key}: deal_rate={deal_rate:.3f} (n={count})")

    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
