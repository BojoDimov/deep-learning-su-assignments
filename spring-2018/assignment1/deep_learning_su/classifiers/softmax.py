import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = np.dot(X[i],W)
    scores -= np.max(scores)
    exp_sum= sum(np.exp(scores))
    correct_score_exp = np.exp(scores[y[i]])
    loss += -np.log(correct_score_exp / exp_sum)
    for j in xrange(W.shape[1]):
        dW[:, j] += X[i] * np.exp(scores[j]) / exp_sum
    dW[:, y[i]] += -X[i]
    
  dW = dW / num_train + 2*reg*W
  loss = loss / num_train + reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  scores -= np.max(scores)
  exp_sum = np.sum(np.exp(scores), axis=1)
  correct_score_exp = np.exp(scores[np.arange(y.shape[0]), y])
  loss = np.sum(-np.log(correct_score_exp / exp_sum)) / X.shape[0] + reg * np.sum(W*W) 
  
  d = np.exp(scores) / exp_sum[:, np.newaxis]
  d[np.arange(y.shape[0]), y] = (correct_score_exp - exp_sum) / exp_sum
  dW = np.dot(X.T, d)
  dW = dW / X.shape[0] + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

