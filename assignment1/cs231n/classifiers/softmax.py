from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = np.shape(W)[1]
    num_train = np.shape(X)[0]

    for i in range(num_train):
      
      #Getting the scores and subtracting high values
      scores = X[i].dot(W)
      scores -= np.max(scores)

      #Getting loss of current example
      loss -= np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))

      #Iterating over all classes for current train example's Weight gradient
      for j in range(num_classes):
        dW[:,j] += np.exp(scores[j]) * X[j] / np.sum(np.exp(scores))
      
      #Removing the correct labels
      dW[:,y[i]] -= X[i]

    #Divide by number of training samples to get mean
    loss /= num_train
    dW /= num_train

    #Adding regularization
    loss += reg * np.sum(W*W)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    #Getting scores and and subtracting high values

    scores = X.dot(W)
    scores -= np.amax(scores, axis=1, keepdims=True)

    #Getting the loss over correct scores
    soft_score = np.exp(scores)/np.sum(np.exp(scores),axis=1, keepdims=True)
    loss = np.sum(-np.log(soft_score[np.arange(num_train),y]))

    #Removing from correct labels 
    soft_score[np.arange(num_train),y] -= 1

    #Calculating gradient
    dW = np.matmul(X.T, soft_score)

    #Dividing by num_train to get average
    loss /= num_train
    dW /= num_train

    #Adding Regularization
    loss += reg * np.sum(W*W)
    dW += reg*2*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
