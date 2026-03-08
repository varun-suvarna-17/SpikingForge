"""
Core SNN components: LIF neuron and base network.
"""

from typing import Optional
import numpy as np


class LIFNeuron:
    """
    Vectorized Leaky Integrate-and-Fire neuron.

    This neuron accumulates `input_current` into its membrane potential. 
    The potential gradually decays over time (leaks). When the potential 
    exceeds the `threshold`, a spike (value `1.0`) is emitted, and the 
    membrane potential is reset to `reset_value`.

    Attributes:
        size (int): The number of neurons.
        threshold (float): The threshold voltage for spike generation.
        decay (float): The factor by which membrane potential decays per timestep.
        reset_value (float): The voltage the membrane is set to after firing.
        v (np.ndarray): State array containing current membrane potentials.
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
        Perform one timestep update for the LIF neuron.

        Args:
            input_current (np.ndarray): The input current for this timestep. 
                                        Must be a NumPy array of shape (size,).

        Returns:
            np.ndarray: Vector containing float spikes (0.0 or 1.0) emitted at this timestep.
            
        Raises:
            ValueError: If the shape of the `input_current` doesn't match the `size` of the neurons.
        """
        if input_current.shape != (self.size,):
            raise ValueError(f"Expected input_current shape ({(self.size,)}), but got {input_current.shape}.")
            
        # Apply membrane decay
        self.v *= self.decay
        
        # Integrate input current
        self.v += input_current
        
        # Generate spikes
        spikes = self.v >= self.threshold
        
        # Reset neurons that fired
        self.v[spikes] = self.reset_value
        
        # Return spike vector as float array
        return spikes.astype(float)

    def reset(self) -> None:
        """
        Reset membrane potential.
        """
        self.v.fill(self.reset_value)