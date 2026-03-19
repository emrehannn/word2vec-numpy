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
 

## positive samples, sliding context over text and yielding target and context pairs. 
# for example, if CONTEXT_SIZE = 2, then for the sentence "the cat sat on the mat", we would yield:
# target: "sat", context: ["the", "cat", "on", "the"]
def get_positive_samples():
    for i in range(CONTEXT_SIZE, len(text_idx) - CONTEXT_SIZE):
        context = np.concatenate((text_idx[i - CONTEXT_SIZE: i], text_idx[i + 1 : i + CONTEXT_SIZE + 1]))
        target = text_idx[i]
        yield target, context

    

## negative sampling

# no. of words occurences
word_counts = np.bincount(text_idx)

noise_dist = word_counts.astype(float) # convert to float for power scaling, otherwise they are integers

noise_dist **= 0.75 # power scaling, as per the word2vec paper to give more weight to less frequent
                    # words and less weight to more frequent words.

noise_dist /= np.sum(noise_dist) # normalize to get probabilities. this is the distribution
                                # we will sample from when generating negative samples.


# negative training pairs
def get_negative_samples(target, k):
    negatives = [] # keep filling negatives until we have k negatives

    while len(negatives) < k:
        sample = np.random.choice(vocabularysize, p=noise_dist)
        if sample != target: # ensure target is not in the negatives.
            negatives.append(sample)
    return np.array(negatives)

# a negative sample is a random word from the vocabulary that is not the target word.
# we will use these negative samples to train our model to distinguish between true context words and random words.
# for example, if the target word is "sat", and the context words are ["the", "cat", "on", "the"],
# we might sample negative words like ["dog", "house", "tree"] that are not in the context of "sat".       