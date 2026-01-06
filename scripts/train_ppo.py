"""Train a PPO agent on Deal or No Deal."""

from __future__ import annotations

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from pyrl_or_no_pyrl import make_batched_env
from pyrl_or_no_pyrl.utils import configure_gpu
from train_common import compute_avg_return


def main() -> None:
    configure_gpu()

    batch_size = 128
    train_py_env = make_batched_env(batch_size=batch_size, seed=21)
    eval_py_env = make_batched_env(batch_size=1, seed=211)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(128, 64),
    )
    value_net = value_network.ValueNetwork(
        train_env.observation_spec(),
        fc_layer_params=(128, 64),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=5,
        train_step_counter=tf.Variable(0),
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=4096,
    )

    for iteration in range(50):
        time_step = train_env.reset()
        for _ in range(128):
            action_step = agent.collect_policy.action(time_step)
            next_time_step = train_env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step

        experience = replay_buffer.gather_all()
        agent.train(experience)
        replay_buffer.clear()

        if iteration % 10 == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=10)
            print(f"Iter {iteration}: avg return = {avg_return:.2f}")


if __name__ == "__main__":
    main()
