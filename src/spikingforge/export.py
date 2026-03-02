"""
Export trained weights to C header format.
"""

import numpy as np
from pathlib import Path


def export_to_c_header(weights: np.ndarray, output_path: str) -> None:
    """
    Export numpy weights as a C header file.
    """
    raise NotImplementedError("Implement C header export logic.")