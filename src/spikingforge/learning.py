"""
Learning rules: STDP and Readout training.
"""

import numpy as np


class STDP:
    """
    Trace-based Spike-Timing Dependent Plasticity.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Update weights using STDP rule.
        """
        raise NotImplementedError("Implement STDP update logic.")


class LinearReadout:
    """
    Linear readout layer trained with SGD.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.randn(input_size, output_size) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def update(self, grad: np.ndarray, lr: float) -> None:
        raise NotImplementedError("Implement SGD update.")