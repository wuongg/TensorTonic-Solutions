import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w ,dtype=float)
    s = np.asarray(s,dtype=float)
    g = np.asarray(g, dtype=float)

    s_t = beta * s + (1 - beta) * g ** 2
    w_t = w - (lr / np.sqrt(s_t + eps)) * g

    return w_t,s_t