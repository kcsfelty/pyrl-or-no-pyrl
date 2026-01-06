"""Train a PPO agent on Deal or No Deal."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

from pyrl_or_no_pyrl import make_batched_env
from pyrl_or_no_pyrl.utils import configure_gpu
from train_common import compute_avg_return


def main() -> None:
    configure_gpu()

    batch_size = 64
    train_py_env = make_batched_env(batch_size=batch_size, seed=21)
    eval_py_env = make_batched_env(batch_size=1, seed=211)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    def _unbatch(spec):
        if hasattr(spec, "minimum"):
            min_val = np.asarray(spec.minimum).reshape(-1)[0]
            max_val = np.asarray(spec.maximum).reshape(-1)[0]
            return tensor_spec.BoundedTensorSpec(
                shape=spec.shape[1:],
                dtype=spec.dtype,
                minimum=min_val,
                maximum=max_val,
                name=spec.name,
            )
        return tensor_spec.TensorSpec(
            shape=spec.shape[1:], dtype=spec.dtype, name=spec.name
        )

    obs_spec = tf.nest.map_structure(_unbatch, train_env.observation_spec())
    action_spec = tf.nest.map_structure(_unbatch, train_env.action_spec())

    class SafeCategoricalProjectionNetwork(network.DistributionNetwork):
        def __init__(
            self,
            sample_spec,
            logits_init_output_factor=0.1,
            name="SafeCategoricalProjectionNetwork",
        ):
            num_actions = int(
                np.asarray(sample_spec.maximum - sample_spec.minimum + 1)
                .reshape(-1)[0]
            )
            if num_actions <= 0:
                raise ValueError("num_actions must be positive.")

            output_shape = sample_spec.shape.concatenate([num_actions])
            output_spec = self._output_distribution_spec(
                output_shape, sample_spec, name
            )

            super().__init__(
                input_tensor_spec=None,
                state_spec=(),
                output_spec=output_spec,
                name=name,
            )

            if not tensor_spec.is_bounded(sample_spec):
                raise ValueError(
                    "sample_spec must be bounded. Got: %s." % type(sample_spec)
                )
            if not tensor_spec.is_discrete(sample_spec):
                raise ValueError(
                    "sample_spec must be discrete. Got: %s." % sample_spec
                )

            self._sample_spec = sample_spec
            self._output_shape = output_shape
            self._projection_layer = tf.keras.layers.Dense(
                self._output_shape.num_elements(),
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=logits_init_output_factor
                ),
                bias_initializer=tf.keras.initializers.Zeros(),
                name="logits",
            )

        def _output_distribution_spec(self, output_shape, sample_spec, network_name):
            input_param_spec = {
                "logits": tensor_spec.TensorSpec(
                    shape=output_shape,
                    dtype=tf.float32,
                    name=network_name + "_logits",
                )
            }

            return distribution_spec.DistributionSpec(
                tfp.distributions.Categorical,
                input_param_spec,
                sample_spec=sample_spec,
                dtype=sample_spec.dtype,
            )

        def call(self, inputs, outer_rank, training=False, mask=None):
            batch_squash = network_utils.BatchSquash(outer_rank)
            inputs = batch_squash.flatten(inputs)
            inputs = tf.cast(inputs, tf.float32)

            logits = self._projection_layer(inputs, training=training)
            logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
            logits = batch_squash.unflatten(logits)

            if mask is not None:
                if mask.shape.rank < logits.shape.rank:
                    mask = tf.expand_dims(mask, -2)
                almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
                logits = tf.compat.v2.where(
                    tf.cast(mask, tf.bool), logits, almost_neg_inf
                )

            return self.output_spec.build_distribution(logits=logits), ()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        obs_spec,
        action_spec,
        fc_layer_params=(128, 64),
        discrete_projection_net=lambda spec: SafeCategoricalProjectionNetwork(spec),
    )
    value_net = value_network.ValueNetwork(
        obs_spec,
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

    for iteration in range(256):
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
