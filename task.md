'''
The task
Given the context words, predict the target:
["the", "cat", ___, "on", "the", "mat"]
                ↑
          what word goes here?

The pipeline
CONTEXT WORDS
"the", "cat", "on", "the"
        ↓
look up their vectors in model matrix (model[context] → pull out 10 rows from model matrix → (10, 100)
        ↓
average them into one vector (hidden layer) (np.mean(..., axis=0) → average across those 10 rows → (100,)
        ↓
multiply against output matrix → get a score for EVERY word in vocab (output @ hidden → dot product against every output row → (253854,)
        ↓
softmax → turn scores into probabilities (softmax(scores) → turn scores into probabilities → (253854,)
        ↓
["sat": 0.40, "banana": 0.001, "walked": 0.23, ...]
        ↓
compare to actual target word → compute loss (-np.log(probs[target]) → how wrong were we? → scalar
        ↓
backpropagate → nudge vectors

averaging        → combine 10 context vectors into 1
output matrix    → "translate" that 1 vector into vocab scores
softmax          → turn raw scores into probabilities that sum to 1
loss             → measure how wrong we were
backprop         → figure out how to be less wrong next time

'''

Task #1

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the ideas behind word2vec, the code, gradient derivations, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.
