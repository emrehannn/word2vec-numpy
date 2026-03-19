# data.py
import numpy as np
from hyperparams import CONTEXT_SIZE

with open("text8", "r") as f:
    text = f.read(1_000_000)


vocabulary = sorted(set(text.split()))

vocabularysize = len(vocabulary)
# 253854

word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

text_idx = [word_to_idx[word] for word in text.split()]

text_idx = np.array(text_idx)
 

## positive samples
def get_positive_samples():
    for i in range(CONTEXT_SIZE, len(text_idx) - CONTEXT_SIZE):
        context = np.concatenate((text_idx[i - CONTEXT_SIZE: i], text_idx[i + 1 : i + CONTEXT_SIZE + 1]))
        target = text_idx[i]
        yield target, context

    

## negative sampling

# no. of words occurences
word_counts = np.bincount(text_idx)

# 3/4 power scaling + normalization from the word2vec paper. this is done because training with softmax is virtually impossible
# (also rare words get a slight boost)
noise_dist = word_counts ** (3/4) / np.sum(word_counts ** (3/4))
                                  #normalization is done so that the distribution sums to 1

# negative training pairs
def get_negative_samples(target, k):
    negatives = [] # keep filling negatives until we have k negatives

    while len(negatives) < k:
        sample = np.random.choice(vocabularysize, p=noise_dist)
        if sample != target: # ensure target is not in the negatives.
            negatives.append(sample)
    return np.array(negatives)
        