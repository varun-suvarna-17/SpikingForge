import sys
from pathlib import Path

# Add src to the Python path so SpikingForge can be imported directly
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from spikingforge.core import LIFNeuron

def run_simulation():
    print("--- SpikingForge LIF Neuron Simulation ---")
    # Create 3 neurons
    neuron = LIFNeuron(size=3, threshold=1.0, decay=0.9, reset_value=0.0)
    
    # Simulate for 5 timesteps with different input currents
    inputs = [
        np.array([0.5, 1.2, 0.0]),
        np.array([0.6, 0.1, 0.5]),
        np.array([0.2, 0.0, 0.6]),
        np.array([0.0, 1.5, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    
    for t, input_curr in enumerate(inputs):
        spikes = neuron.forward(input_curr)
        print(f"Time {t+1}:")
        print(f"  Input Current:      {input_curr}")
        print(f"  Membrane Potential: {neuron.v}")
        print(f"  Spikes:             {spikes}\n")

if __name__ == "__main__":
    run_simulation()
