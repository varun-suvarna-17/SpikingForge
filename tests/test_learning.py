import numpy as np
from SpikingForge.learning import STDP, LinearReadout, sgd_step


def test_stdp_update():

    stdp = STDP(3, 2)

    W = np.random.rand(2, 3)

    pre = np.array([1, 0, 1])
    post = np.array([0, 1])

    W_new = stdp.update(W.copy(), pre, post)

    assert W_new.shape == (2, 3)


def test_linear_readout_forward():

    model = LinearReadout(4, 2)

    x = np.random.rand(4)

    y = model.forward(x)

    assert y.shape == (2,)


def test_sgd_step():

    model = LinearReadout(4, 2)

    x = np.random.rand(4)
    y_true = np.array([1.0, 0.0])

    error = sgd_step(model, x, y_true)

    assert error.shape == (2,)