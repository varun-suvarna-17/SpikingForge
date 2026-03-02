"""
Core SNN components: LIF neuron and base network.
"""

from typing import Optional
import numpy as np


class LIFNeuron:
    """
    Vectorized Leaky Integrate-and-Fire neuron.
    """

    def __init__(
        self,
        size: int,
        threshold: float = 1.0,
        decay: float = 0.9,
        reset_value: float = 0.0,
    ) -> None:
        self.size = size
        self.threshold = threshold
        self.decay = decay
        self.reset_value = reset_value

        self.v = np.zeros(size)  # membrane potential

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """
        Perform one timestep update.
        """
        raise NotImplementedError("Implement LIF forward logic.")

    def reset(self) -> None:
        """
        Reset membrane potential.
        """
        self.v.fill(self.reset_value)