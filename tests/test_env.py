import numpy as np

from pyrl_or_no_pyrl.env import DealOrNoDealEnv


def test_deal_ends_episode():
    env = DealOrNoDealEnv(seed=1)
    time_step = env.reset()
    assert not time_step.is_last()
    time_step = env.step(0)
    assert time_step.is_last()
    assert time_step.reward >= 0.0


def test_no_deal_reaches_terminal():
    env = DealOrNoDealEnv(seed=2)
    time_step = env.reset()
    for _ in range(100):
        time_step = env.step(1)
        if time_step.is_last():
            break
    assert time_step.is_last()
    assert time_step.reward >= 0.0


def test_offer_within_range():
    env = DealOrNoDealEnv(seed=3)
    time_step = env.reset()
    obs = time_step.observation
    offer = obs[-1]
    assert offer >= 0.0
    assert offer <= np.max(env._values)
