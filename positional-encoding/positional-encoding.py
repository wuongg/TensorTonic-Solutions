import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    arr = np.zeros((seq_len,d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            exponent = 2 * (i // 2) / d_model
            angle = pos / (base ** exponent)
            if i % 2 == 0:
                arr[pos,i] = np.sin(angle)
            else:
                arr[pos,i]= np.cos(angle)
    return arr
            
    