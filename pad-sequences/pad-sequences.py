import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    N = len(seqs)
    
    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len
    
    padded = []
    
    for seq in seqs:
        new_seq = list(seq[:L])  # truncate nếu cần
        pad_length = L - len(new_seq)
        if pad_length > 0:
            new_seq += [pad_value] * pad_length
        padded.append(new_seq)
    
    return np.array(padded)