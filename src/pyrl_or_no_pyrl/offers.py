"""Offer curve utilities."""

from __future__ import annotations

import numpy as np


def banker_offer(remaining_values: np.ndarray, step: int, max_steps: int) -> float:
    """Compute a simple banker offer based on remaining values.

    The offer ramps from 60% to 100% of expected value as the game progresses.
    """
    expected = float(np.mean(remaining_values))
    if max_steps <= 0:
        return expected
    progress = min(max(step / max_steps, 0.0), 1.0)
    multiplier = 0.6 + 0.4 * progress
    return expected * multiplier
