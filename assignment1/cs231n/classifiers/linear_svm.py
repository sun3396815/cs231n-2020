from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # scores: (C,) = (D,).dot(D, C)
        scores = X[i].dot(W)
        # result of w·x
        correct_class_score = scores[y[i]]
        num_classes_greater_than_zero = 0
        for j in range(num_classes):
            # y[i] is the ground truth class, between [0,C-1]
            if j == y[i]:
                continue
            # hinge loss
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                num_classes_greater_than_zero += 1
                loss += margin
                dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] * num_classes_greater_than_zero

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (N,C) = (N,D)·(D,C)
    scores = np.dot(X, W)
    C = W.shape[1]
    N = X.shape[0]
    losses = np.zeros(scores.shape)
    #(N,), np.choose 根据index array从ndarray中提取元素 或者可以写成 gts = scores[range(N), y]
    gts = np.choose(y, scores.T)
    losses = scores - gts.reshape((N, 1)) + 1
    losses[losses < 0] = 0
    # fixme 注意这里的写法！,不可以把range(N)写成":", ground truth对应的class，loss为0
    losses[range(N),y] = 0

    loss = np.sum(losses)
    loss /= N
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #(N,C)
    mask = (losses > 0).astype(float)
    #(N,)
    num_margin_greater_than_zero = np.sum(mask, axis=1)
    # fixme: ground truth对应位置 减去 num_margin_greater_than_zero
    mask[range(num_margin_greater_than_zero.shape[0]), y] -= num_margin_greater_than_zero

    # dW: (D,C), X: (N,D), losses: (N,C)
    dW += np.dot(X.T, mask)
    dW = dW / N + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
