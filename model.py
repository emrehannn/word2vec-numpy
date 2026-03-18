# model.py
import numpy as np
from hyperparams import EMBEDDING_DIM, LEARNING_RATE, CONTEXT_SIZE
from data import vocabularysize


# embedding matrices

W_in = np.random.randn(vocabularysize, EMBEDDING_DIM)
W_out = np.random.rand(vocabularysize, EMBEDDING_DIM)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# forward pass

# hidden layer: h = mean of context picked from W_in
# scores: W_out @ h
# probabilities: softmax of scores

def forward_pass(context, target):
    hidden_layer = np.mean(W_in[context], axis=0)
    target_embedding = W_out[target]

    # dot product of hidden layer and target embedding
    score = np.dot(W_out, hidden_layer)
    # softmax score of every word in the vocabulary
    probability = softmax(score)
    return probability, hidden_layer, target_embedding

# compute loss

def compute_loss(probability, target):
    loss = -np.log(probability[target])
    return loss


# backward pass

#  backwards from forward pass, chain rule. To see how much a weight (w) affected the total error,
# look at how much the weights affected the score, and then how much that score affected the error

# dL/dW = dL/dS x dS/dW


def backward_pass(probability, target, hidden_layer, context):
    loss = compute_loss(probability, target)
    # step 1  w.r.t scores dL/dS
    one_hot_target = np.zeros(vocabularysize)
    one_hot_target[target] = 1
    d_scores_loss = probability - one_hot_target # how wrong was each probability


    #step 2 w.r.t output layer dL/dW_out = dS x h[j]
    d_output_layer_loss = np.outer(d_scores_loss, hidden_layer) # 
  

    #step 3 derivatife of loss w.r.t. hidden layer same as above,
    #  but we blame hidden instead of output layer for the error. dL/dh = W_out.T @ d_scores
    d_hidden = np.dot(W_out.T , d_scores_loss)

    #step 4 derivatife of loss w.r.t. input layer loss averaged over to hidden/size dL/(dW_in[context])
    W_in[context] -= LEARNING_RATE * (d_hidden / CONTEXT_SIZE)
    
    W_out -= d_output_layer_loss * LEARNING_RATE


# ── FORWARD (what we computed) ──────────────────────────────
# hidden   = mean(W_in[context])
# scores   = W_out @ hidden
# probs    = softmax(scores)
# loss     = -log(probs[target])

# ── BACKWARD (reverse order, chain rule) ────────────────────

# step 1 — loss → scores
# how wrong was each score?
# d_scores = probs - onehot(target)

# step 2 — scores → W_out
# how much did each output weight cause that wrongness?
# d_W_out = outer(d_scores, hidden)

# step 3 — scores → hidden
# how much did the hidden layer cause that wrongness?
# d_hidden = W_out.T @ d_scores

# step 4 — hidden → W_in
# how much did each input word embedding cause that hidden error?
# d_W_in = d_hidden / context_size   (divide because forward was a mean)