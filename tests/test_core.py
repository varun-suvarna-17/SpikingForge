import pytest
import numpy as np
from spikingforge.core import LIFNeuron


def test_lif_initialization():
    neuron = LIFNeuron(size=10)
    assert neuron.v.shape[0] == 10

def test_spike_generation():
    """Test 1 - Spike Generation: Input above threshold should produce a spike."""
    neuron = LIFNeuron(size=1, threshold=1.0, decay=0.9, reset_value=0.0)
    input_current = np.array([1.5])
    spikes = neuron.forward(input_current)
    assert np.array_equal(spikes, np.array([1.0]))

def test_no_spike():
    """Test 2 - No Spike: Input below threshold should not spike."""
    neuron = LIFNeuron(size=1, threshold=1.0, decay=0.9, reset_value=0.0)
    input_current = np.array([0.5])
    spikes = neuron.forward(input_current)
    assert np.array_equal(spikes, np.array([0.0]))
    
def test_reset_behavior():
    """Test 3 - Reset Behavior: Ensure membrane potential resets after spike."""
    neuron = LIFNeuron(size=1, threshold=1.0, decay=0.9, reset_value=0.0)
    input_current = np.array([1.5])
    neuron.forward(input_current)
    assert neuron.v[0] == 0.0  # Should be reset to reset_value
    
def test_output_shape():
    """Test 4 - Output Shape: Output should match neuron size."""
    neuron = LIFNeuron(size=5, threshold=1.0, decay=0.9, reset_value=0.0)
    input_current = np.array([0.0, 1.2, 0.5, 1.5, 0.2])
    spikes = neuron.forward(input_current)
    assert spikes.shape == (5,)

def test_input_shape_validation():
    """Test to ensure informative error is raised for shape mismatch."""
    neuron = LIFNeuron(size=3)
    input_current = np.array([1.0, 2.0]) # Shape is (2,) instead of (3,)
    with pytest.raises(ValueError):
        neuron.forward(input_current)