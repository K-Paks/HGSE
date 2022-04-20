import numpy as np

def dot_score(a, b):
    if len(a.shape) == 1:
        a = np.expand_dims(a, 0)

    if len(b.shape) == 1:
        b = np.expand_dims(b, 0)

    return np.matmul(a, b.T)