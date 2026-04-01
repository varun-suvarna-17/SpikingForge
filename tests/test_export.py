import numpy as np
import tempfile
import os
from Spikingforge.Export import export_to_c_header

def test_export_to_c_header_creates_file():
    # Create dummy weights
    weights = {
        'stdp_weights': np.random.rand(784, 100).astype(np.float32),
        'readout_weights': np.random.rand(100, 10).astype(np.float32),
        'readout_bias': np.random.rand(10).astype(np.float32)
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.h', delete=False) as tmp:
        export_to_c_header(weights, tmp.name)
        tmp.flush()

        # Check file exists and has content
        assert os.path.exists(tmp.name)
        with open(tmp.name, 'r') as f:
            content = f.read()

        # Verify essential parts
        assert '#ifndef SPIKINGFORGE_WEIGHTS_H' in content
        assert '#define INPUT_SIZE 784' in content
        assert '#define HIDDEN_SIZE 100' in content
        assert '#define OUTPUT_SIZE 10' in content
        assert 'static const float stdp_weights[784][100]' in content
        assert 'static const float readout_weights[100][10]' in content
        assert 'static const float readout_bias[10]' in content
        assert '#endif' in content

    os.unlink(tmp.name)

def test_export_to_c_header_formatting():
    # Test with small arrays to check formatting
    weights = {
        'stdp_weights': np.array([[0.123456789, 0.234567891]], dtype=np.float32),
        'readout_weights': np.array([[0.345678912, 0.456789123]], dtype=np.float32),
        'readout_bias': np.array([0.567891234, 0.678912345], dtype=np.float32)
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.h', delete=False) as tmp:
        export_to_c_header(weights, tmp.name)
        with open(tmp.name, 'r') as f:
            content = f.read()

        # Check that numbers are formatted to 6 decimal places (rounding may cause slight variations)
        assert '0.123457' in content or '0.123456' in content
        assert '0.234568' in content
        assert '0.345679' in content
        assert '0.456789' in content
        assert '0.567891' in content
        assert '0.678912' in content

    os.unlink(tmp.name)