import math

import numpy as np
import scipy as sp
import functools

def calculate_l(D):
    """
    Calculates the 'L' constant for FISTA for the dictionary D

    Parameters
    D : array
    """
    N_sp = D.shape[0]
    try:
        L = 2 * sp.linalg.eigh(np.dot(D, D.T), 
                               eigvals_only=True, 
                               eigvals=(N_sp-1,N_sp-1))[0]
    except ValueError:
        L = (2 * D.shape[1])
    return L

def fista(cost, grad_A, A0, n_g_steps, lamb, l, 
          track_costs=False, pos_only=True):
    """
    Computes FISTA Approximation

    Parameters
    ----------
    cost : function handle
        Function handle that gives the cost given the sparse coefficients, A
    grad_A: function handle
        Function that takes in the sparse coefficients 
        and gives dE_reconstruction_error/dA
    A0 : matrix
        Initial value for sparse coefficents
    n_g_steps : int
        Number of gradient steps
    l : float
        Lipschitz constant for grad_A
    track_costs : bool
    pos_only : bool
        Positive only sparse coding
    """
    n_bat, n_sp = A0.shape
    Xs = np.zeros((2, n_bat, n_sp))
    Ys = np.zeros((2, n_bat, n_sp))
    Ts = np.zeros((2,)).astype('float32')
    if track_costs:
        costs = np.zeros(n_g_steps-1)
    Xs[0] = A0
    Ys[0] = A0
    Ts[0] = 1.
    for i in range(n_g_steps-1):
        i0 = i % 2
        i1 = (i+1) % 2
        A = Ys[i0]
        Xs[i1] = ista(A, grad_A(A), lamb, l, pos_only)
        Ts[i1] = 0.5 * (1. + np.sqrt(1. + 4. * Ts[i0] ** 2))
        Ys[i1] = Xs[i1] + (Ts[i0]-1)/ Ts[i1] * (Xs[i1]-Xs[i0])
        E = cost(A)
        if track_costs:
            costs[i] = E
        if (i % (n_g_steps/10) == 0) and track_costs:
            print E
    if track_costs:
        rval = (Xs[(n_g_steps-1)%2], costs)
    else:
        rval = Xs[(n_g_steps-1)%2] 
    return rval

def ista(A, gradA, lamb, l, pos_only=True):
    """
    Returns the new value of the sparse coefficients after one ista step
        for the cost function f(A) + lamb * |A|
    Parameters
    ----------
    A : array
        Current value of sparse coefficients
    gradA : array
        Derivative of df(A)/dA
    alpha : float
        Parameter for regularization term
    l : float
        Lipschitz constant for dE_rec/dA
    pos_only : bool
        Positive only sparse coding
    """
    theta = lamb / l
    Ap = A - 1./l * gradA
    tAp = np.sign(Ap) * np.maximum(np.abs(Ap) - theta, 0.)
    if pos_only:
        tAp = np.maximum(tAp, 0.)
    return tAp

def row_pos_normalize(D):
    D = np.maximum(D, 0.000001)
    return row_normalize(D)

def row_normalize(D):
    norms = np.maximum(np.sqrt((D ** 2).sum(1, keepdims=True)), 
                       0.0000001)
    D = D * 1./norms
    return D

class Network(object):
    """
    Class which represents a sparse coding network.

    Parameters
    ----------
    n_dict : int
        Number of dictionary elements.
    n_features : int
        Number (dimensionality) of features (eg. number of pixels)
    lamb : float
        Sparsity coefficient.
    eta : float
        Dictionary learning rate.
    pos_only : bool
        True if we want all dictionary elements and coefficients to be
        positive numbers
    """
    def __init__(self, n_dict, n_features, lamb=.1, eta=.01, 
                 batch_size = 100, pos_only = True):
        self.n_dict = n_dict
        self.n_features = n_features
        self.lamb = lamb
#        self.eta = eta
        self.batch_size = batch_size
        self.pos_only = pos_only
        self.reset()
        self.stale_A = True

    def reset(self, n_dict=None, n_features=None):
        """
        Reset dictionary to new sizes or default to old sizes.

        Parameters
        ----------
        n_dict : int (optional)
        n_features : int (optional)
        """
        if n_dict is None:
            n_dict = self.n_dict
        if n_features is None:
            n_features = self.n_features
        # Fix for positive only
        self.D = row_normalize(np.random.rand(n_dict,
                                            n_features))

    def infer_A(self, X, A0=None, n_g_steps=40, track_cost=False):
        """
        Infer sparse coefficients, A.

        Parameters
        ----------
        X : array
            A batch of input data.
        A0 : array
            Starting values for A
        n_g_steps : int
            Number of gradient steps for inference.
        track_cost : bool
            True if we track cost, then return from fista is a tuple
        """
        self.stale_A = False
        self.X = X
        if A0 is None:
            A0 = np.zeros((self.batch_size, self.n_dict))
        cost = functools.partial(self.cost, X)
        grad = functools.partial(self.grad_A, X)
        l = calculate_l(self.D)
        A = fista(cost, grad, A0, n_g_steps, self.lamb, l, 
                  track_cost, self.pos_only)
        if isinstance(A, tuple):
            self.A, cost = A
        else:
            self.A = A
        return A

    def grad_A(self, X, A):
        """
        Gradient of the reconstruction error w.r.t. A.

        Parameters
        ----------
        X : array
            A batch of input data.
        A : array 
            Sparse coefficients.
        """
        return -(X-A.dot(self.D)).dot(self.D.T)

    def grad_D(self, X, A):
        """
        Gradient of the reconstruction error w.r.t. D.

        Parameters
        ----------
        X : array
            Input data.
        A : array 
            Sparse coefficients.
        """
        return -(A.T).dot(X-A.dot(self.D))

    def learn_D(self, X=None, A=None, eta = 0.01):
        """
        Run one step of dictionary learning on D.

        Parameters
        ----------
        X : array (optional)
            Input data.
        A : array  (optional)
            Sparse coefficients.
        eta : double
            Learning rate for the dictionary
        """
        if self.stale_A and A is None:
            raise AttributeError("Coefficients: A, are old. "+
                                 "Run infer_A before learn_D")
        if X is None:
            X = self.X
        if A is None:
            A = self.A
        self.D = row_pos_normalize(self.D-eta*
                               row_normalize(self.grad_D(X, A)))
                               #self.grad_D(X, A))
        self.stale_A = True

    def train(self, data, batch_size=100, n_epochs=10, eta = 0.05,
              reset=True, rng=None):
        """
        Train a dictionary: D, on the data: X.

        Parameters
        ----------
        data : array
            Input data. (n_examples, n_features)
        batch_size : int  (optional)
            Batch size for training.
        n_epochs : int (optional)
            Number of passes through dataset (epochs)
        eta : double
            Step size for dictionary learning
        reset : boolean (optional)
            Reset dictionary before training.
        """
        if rng is None:
            rng = np.random
        if reset:
            self.reset()
        n_examples = data.shape[0]
        if len(data.shape) > 2:
            data = data.reshape(n_examples, -1)
        batch_size = min(batch_size, n_examples)
        n_batches = int(math.ceil(n_examples/batch_size))
        for ii in range(n_epochs):
            order = np.random.permutation(n_batches)
            for jj in order:
                batch = data[jj*batch_size:min((jj+1)*batch_size, n_examples)]
                self.X = batch
                A = self.infer_A(batch)
                self.learn_D(eta=eta)
            print("Epoch "+str(ii+1)+" of "+str(n_epochs))
            print("MSE:      "+str(self.MSE(batch, A)))
            print("Sparsity: "+str(self.sparsity(A)))
            print("SNR:      "+str(self.SNR(batch, A)))
            print("Cost:     "+str(self.cost(batch, A)))
            print

    def reconstruct(self, X=None, A=None):
        """
        Reconstruct the data from the learned model.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        if X is None:
            X = self.X
        if A is None:
            A = self.infer_A(X)
        return A.dot(self.D)

    def MSE(self, X, A):
        """
        One-half mean-squared error for reconstructed data.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        X_prime = self.reconstruct(X, A) 
        return .5*((X-X_prime)**2).mean(0).sum()

    def SNR(self, X, A=None):
        """
        Signal-to-noise ration for reconstructed data.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        X_prime = self.reconstruct(X, A=A)
        error = X-X_prime
        return (X.var(0)/error.var(0)).mean()

    def sparsity(self, A=None):
        """
        Sparsity measure for coefficients.

        Parameters
        ----------
        A : array (optional)
            Sparse coefficients.
        """
        if A is None:
            assert X is not None, 'Must provide A or X'
            A = self.infer_A(X)
        return self.lamb*np.absolute(A).mean(0).sum()

    def cost(self, X, A):
        """
        Sparse coding cost.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        return self.MSE(X, A)+self.sparsity(A)
