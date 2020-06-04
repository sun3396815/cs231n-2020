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

    num_train = X.shape[0]
    dim = X.shape[1]

    for i in range(num_train):
        # x[i]: (D,) W: (D, C) scores: (C,)
        scores = np.dot(X[i], W)
        exps = np.exp(scores)
        softmax = exps / np.sum(exps)

        # softmax的损失函数为sum(-ti*log(yi)), 除开真实分类为1，其他均为0，所以简化为-log(yi), yi为真实分类对应的softmax值
        loss -= np.log(softmax[y[i]])
        # https://zhuanlan.zhihu.com/p/27223959 这里介绍了softmax损失函数及导数的公式推导
        # loss对每个分类的softmax值求导为：yi - ti, 所以真实分类的softmax值减去1，其余不变。
        # 再对参数w求导，有参数的梯度：(yi - ti) · X
        # dW:(D,C) d:(C,) X(D)
        softmax[y[i]] = softmax[y[i]] - 1
        d = np.dot(X[i].reshape(-1, 1), softmax.reshape(1, -1))
        dW += d

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

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

    # scores : (N, C)
    num_train = X.shape[0]
    scores = np.dot(X, W)
    exps = np.exp(scores)
    softmaxs = exps / np.sum(exps, axis=1).reshape(-1, 1)
    loss = np.mean(-np.log(softmaxs[range(softmaxs.shape[0]), y])) + reg * np.sum(W * W)

    # now softmax is the derivative, (N, C)
    softmaxs[range(softmaxs.shape[0]), y] = softmaxs[range(softmaxs.shape[0]), y] - 1
    # X (N, D)
    dW = np.dot(X.T, softmaxs) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
