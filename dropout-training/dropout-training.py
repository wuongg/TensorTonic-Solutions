import numpy as np

def dropout(x, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    keep_prob = 1.0 - p

    # raw mask 0/1
    mask = (rng.random(x.shape) < keep_prob).astype(float)

    # scale mask (inverted dropout)
    scaled_mask = mask / keep_prob

    output = x * scaled_mask

    return output, scaled_mask