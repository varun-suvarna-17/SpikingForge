import numpy as np
from spikingforge.core import LIFNeuron


def test_lif_initialization():
    neuron = LIFNeuron(size=10)
    assert neuron.v.shape[0] == 10