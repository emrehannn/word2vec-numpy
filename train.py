#train.py
from model import Word2Vec
from data import get_positive_samples, get_negative_samples
from hyperparams import EPOCHS, k
import numpy as np


model = Word2Vec()


for epoch in range(EPOCHS):
    epoch_loss = 0
    pairs = 0
    for target, context in get_positive_samples():
        negatives = get_negative_samples(target, k)

        target_score, negative_scores = model.forward_pass(context, target, negatives)
        model.backward_pass(target, target_score, negative_scores, negatives, context)
        epoch_loss += model.compute_loss(target_score, negative_scores)
        
        pairs += 1
    average_loss = epoch_loss / pairs
    
    print(f"Epoch {epoch} loss: {average_loss}")
    
    

np.save("embeddings.npy", model.W_in)

