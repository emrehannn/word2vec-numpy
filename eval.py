import numpy as np
from data import word_to_idx

E = np.load("embeddings.npy")
E = E / np.linalg.norm(E, axis=1, keepdims=True)

def sim(a, b):
    return E[word_to_idx[a]] @ E[word_to_idx[b]]

print(sim("king", "queen"))   # should be high
print(sim("king", "dog"))     # should be low

# test embeddings, king, queen, france, paris   

