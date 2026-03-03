import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)
    if x.shape != p.shape:
        raise ValueError("Shapes of x and p must match.")
    
    # Ensure probabilities sum to 1 within tolerance
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1 within tolerance 1e-6.")
    return np.sum(x*p)
