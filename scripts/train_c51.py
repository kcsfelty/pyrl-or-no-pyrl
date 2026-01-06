"""Train a C51 (categorical DQN) agent on Deal or No Deal."""

from __future__ import annotations

import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from pyrl_or_no_pyrl import make_batched_env
from pyrl_or_no_pyrl.utils import configure_gpu
from train_common import collect_step, compute_avg_return


def main() -> None:
    configure_gpu()

    batch_size = 64
    train_steps = 256
    train_py_env = make_batched_env(batch_size=batch_size, seed=11)
    eval_py_env = make_batched_env(batch_size=1, seed=111)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(128, 64),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    epsilon_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1.0,
        decay_steps=train_steps,
        decay_rate=0.01,
        staircase=False,
    )
    train_step_counter = tf.Variable(0)
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE
        ),
        epsilon_greedy=(lambda: epsilon_schedule(train_step_counter)),
        gamma=1.0,
        train_step_counter=train_step_counter,
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=5000,
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    for _ in range(200):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    for step in range(train_steps):
        collect_step(train_env, agent.collect_policy, replay_buffer)
        experience, _ = next(iterator)
        agent.train(experience)
        if step % 200 == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=10)
            print(f"Step {step}: avg return = {avg_return:.2f}")


if __name__ == "__main__":
    main()
