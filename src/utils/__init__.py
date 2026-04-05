import os
import random

import numpy as np
import tensorflow as tf


def project_root() -> str:
    """Return the repository root based on this file location."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_global_seed(seed: int = 42) -> None:
    """Set seed values for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
