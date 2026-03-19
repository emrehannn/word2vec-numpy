# model.py
import numpy as np
from hyperparams import EMBEDDING_DIM, LEARNING_RATE
from data import vocabularysize


class Word2Vec:
    def __init__(self):
        self.W_in = np.random.randn(vocabularysize, EMBEDDING_DIM)     # input embeddings, one vector per word in the vocab
        self.W_out = np.random.randn(vocabularysize, EMBEDDING_DIM)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, context, target, negatives):

        # mean the context vectors into one in the Y axis. (2*context , embeddingdim) to (embeddingdim, )
        self.hidden = np.mean(self.W_in[context], axis = 0)

        # dot output matrix with hidden dimension values to get the score, both are (100, )
        target_score = self.sigmoid(np.dot(self.W_out[target], self.hidden))

        # its the same, but there are k negative scores and shape is thus (k, )
        negative_scores = self.sigmoid(self.W_out[negatives] @ self.hidden)

        return target_score, negative_scores


    def compute_loss(self, target_score, negative_scores):
        # L = -log(target_score) - sum(log(1 - negative_scores))
        return -np.log(target_score) - np.sum(np.log(1 - negative_scores))
        # -positive signal - negative signal



    def backward_pass(self, target, target_score, negative_scores, negatives, context):

        
        target_error = target_score - 1
        d_W_out = target_error * self.hidden
        self.W_out[target] -= LEARNING_RATE * d_W_out
        

        negative_error = negative_scores
        d_W_out_negatives = np.outer(negative_error, self.hidden) # only outer product gives (k, EMBEDDING_DIM) ...
        self.W_out[negatives] -= LEARNING_RATE * d_W_out_negatives


        d_hidden = target_error * self.W_out[target] + negative_error @ self.W_out[negatives]

        self.W_in[context] -= LEARNING_RATE * (d_hidden / len(context))


