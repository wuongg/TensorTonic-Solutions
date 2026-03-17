import numpy as np

def streaming_minmax_init(D):
    return {
        'min': np.full(D, np.inf),
        'max': np.full(D, -np.inf)
    }
def streaming_minmax_update(state, X_batch, eps=1e-8):
    X_batch = np.asarray(X_batch)
    
    # Step 1: batch stats
    batch_min = X_batch.min(axis=0)
    batch_max = X_batch.max(axis=0)
    
    # Step 2: update global stats
    state['min'] = np.minimum(state['min'], batch_min)
    state['max'] = np.maximum(state['max'], batch_max)
    
    # Step 3: normalize using updated stats
    denom = state['max'] - state['min']
    X_norm = (X_batch - state['min']) / (denom + eps)
    
    return X_norm