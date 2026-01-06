"""Utility helpers."""

from __future__ import annotations

import tensorflow as tf


def configure_gpu() -> None:
    """Enable GPU memory growth if a GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
