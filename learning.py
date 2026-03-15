import numpy as np


class STDP:
    """
    Trace-based Spike Timing Dependent Plasticity (STDP)

    Parameters
    ----------
    n_pre : int
        Number of presynaptic neurons
    n_post : int
        Number of postsynaptic neurons
    lr : float
        Learning rate
    tau_pre : float
        Pre-synaptic trace decay
    tau_post : float
        Post-synaptic trace decay
    w_min : float
        Minimum weight
    w_max : float
        Maximum weight
    """

    def __init__(
        self,
        n_pre,
        n_post,
        lr=0.01,
        tau_pre=20.0,
        tau_post=20.0,
        w_min=0.0,
        w_max=1.0,
    ):

        self.lr = lr
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_min = w_min
        self.w_max = w_max

        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)

    def update(self, W, pre_spikes, post_spikes):
        """
        Update weights using STDP rule.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix (n_post, n_pre)
        pre_spikes : np.ndarray
            Binary vector of presynaptic spikes
        post_spikes : np.ndarray
            Binary vector of postsynaptic spikes
        """

        # decay traces
        self.pre_trace *= np.exp(-1 / self.tau_pre)
        self.post_trace *= np.exp(-1 / self.tau_post)

        # add new spikes
        self.pre_trace += pre_spikes
        self.post_trace += post_spikes

        # LTP (pre before post)
        dW_plus = self.lr * np.outer(post_spikes, self.pre_trace)

        # LTD (post before pre)
        dW_minus = self.lr * np.outer(self.post_trace, pre_spikes)

        W += dW_plus
        W -= dW_minus

        # clip weights (numerical stability)
        np.clip(W, self.w_min, self.w_max, out=W)

        return W
    
class LinearReadout:
    """
    Simple linear readout layer for spike features.
    """

    def __init__(self, in_features, out_features):

        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        """
        Forward pass.

        x : shape (in_features,)
        """
        return self.W @ x + self.b

def sgd_step(model, x, y_true, lr=0.01):
    """
    Perform one SGD update step.

    Parameters
    ----------
    model : LinearReadout
    x : np.ndarray
    y_true : np.ndarray
    lr : float
    """

    y_pred = model.forward(x)

    # gradient of MSE
    error = y_pred - y_true

    grad_W = np.outer(error, x)
    grad_b = error

    model.W -= lr * grad_W
    model.b -= lr * grad_b

    return error