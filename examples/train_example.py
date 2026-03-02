import numpy as np

from spikingforge.core import LIFNeuron
from spikingforge.learning import LinearReadout


def main():
    # Create dummy input
    input_size = 10
    output_size = 2

    neuron = LIFNeuron(size=input_size)

    # Simulate one timestep
    input_current = np.random.rand(input_size)
    spikes = neuron.forward(input_current)

    print("Spikes:", spikes)

    # Test readout layer
    readout = LinearReadout(input_size, output_size)
    output = readout.forward(spikes)

    print("Readout output:", output)


if __name__ == "__main__":
    main()