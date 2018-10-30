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
  nums_train,_ = X.shape
  _,nums_cls = W.shape  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dscore = np.zeros(shape = (nums_train,nums_cls))
  for i in range(nums_train):
    score = np.dot(X[i],W)
    shift_score = score - np.max(score)
    loss += -np.log(np.exp(shift_score[y[i]])/np.sum(np.exp(shift_score)))
    dscore[i] = np.exp(shift_score)/np.sum(np.exp(shift_score))
    dscore[i,y[i]] -= 1  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/nums_train + reg*np.sum(W*W)
  dW = np.dot(X.T,dscore)/nums_train + reg*2*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  nums_train,_ = X.shape
  _,nums_cls = W.shape 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.dot(X,W)
  max_score = np.max(score,axis = 1)
  shift_score = score - max_score.reshape(-1,1)
  prob = np.exp(shift_score)/np.sum(np.exp(shift_score),axis = 1).reshape(-1,1)
  #one hot coding 
  eye = np.eye(W.shape[1])
  categorical_y = eye[y]
  loss = np.sum(-np.log(prob)[range(nums_train),y])/nums_train + reg*np.sum(W*W)
  #此处有坑np.log(prob).dot(categorical_y.T) :the result is a matrix
  dscore = prob - categorical_y
  dW = np.dot(X.T,dscore)/nums_train + reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

