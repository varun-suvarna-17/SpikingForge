"""
Export trained weights to C header format.
"""

import numpy as np
from pathlib import Path


def export_to_c(weights_dict: dict, output_file: str) -> None:
    """
    Export trained weights to a C header file.

    Args:
        weights_dict: Dictionary with keys:
            - 'stdp_weights': numpy array of shape (input_size, hidden_size)
            - 'readout_weights': numpy array of shape (hidden_size, output_size)
            - 'readout_bias': numpy array of shape (output_size,)
        output_file: Path to the output .h file.
    """
    stdp_w = weights_dict['stdp_weights']
    readout_w = weights_dict['readout_weights']
    readout_b = weights_dict['readout_bias']

    input_size, hidden_size = stdp_w.shape
    output_size = readout_b.shape[0]

    with open(output_file, 'w') as f:
        f.write("// SpikingForge exported weights\n")
        f.write("// Generated automatically - do not edit\n\n")
        f.write("#ifndef SPIKINGFORGE_WEIGHTS_H\n")
        f.write("#define SPIKINGFORGE_WEIGHTS_H\n\n")
        f.write(f"#define INPUT_SIZE {input_size}\n")
        f.write(f"#define HIDDEN_SIZE {hidden_size}\n")
        f.write(f"#define OUTPUT_SIZE {output_size}\n\n")

        # STDP weights
        f.write("// Input to hidden weights (STDP)\n")
        f.write(f"static const float stdp_weights[{input_size}][{hidden_size}] = {{\n")
        for i in range(input_size):
            f.write("    {")
            for j in range(hidden_size):
                f.write(f"{stdp_w[i, j]:.6f}")
                if j < hidden_size - 1:
                    f.write(", ")
            f.write("}")
            if i < input_size - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")

        # Readout weights
        f.write("// Hidden to output weights (readout)\n")
        f.write(f"static const float readout_weights[{hidden_size}][{output_size}] = {{\n")
        for i in range(hidden_size):
            f.write("    {")
            for j in range(output_size):
                f.write(f"{readout_w[i, j]:.6f}")
                if j < output_size - 1:
                    f.write(", ")
            f.write("}")
            if i < hidden_size - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")

        # Readout bias
        f.write("// Readout bias\n")
        f.write(f"static const float readout_bias[{output_size}] = {{")
        for j in range(output_size):
            f.write(f"{readout_b[j]:.6f}")
            if j < output_size - 1:
                f.write(", ")
        f.write("};\n\n")

        f.write("#endif // SPIKINGFORGE_WEIGHTS_H\n")