import math

import numpy as np
import scipy as sp

def calculate_L(D):
    """
    Calculates the 'L' constant for FISTA for the dictionary D

    Parameters
    D : array
    """
    try:
        L = 2 * sp.linalg.eigh(np.dot(D, D.T), 
                               eigvals_only=True, 
                               eigvals=(N_sp-1,N_sp-1))[0]
    except ValueError:
        L = (2 * D.shape[1])
    return L

def fista(cost, grad_A, A0, n_steps, l, print_costs = False):
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
    n_steps : int
        Number of iterations
    l : float
        Lipschitz constant for grad_A
    """
    n_bat, n_sp = A0.shape
    Xs = np.zeros((2, n_bat, n_sp))
    Ys = np.zeros((2, n_bat, n_sp))
    Ts = np.zeros((2,)).astype('float32')
    Xs[0] = A0
    Ys[0] = A0
    Ts[0] = 1.
    for i in range(n_steps-1):
        i0 = i % 2
        i1 = i % 2
        A = Ys[i0]
        Xs[i1] = ista(A, grad_A(A),  alpha, l)
        Ts[i1] = 0.5 * (1. + np.sqrt(1. + 4. * Ts[i0] ** 2))
        Ys[i1] = Xs[i1] + (Ts[i0]-1)/ Ts[i1] * (Xs[i1]-Xs[i0])
        E, E_rec, E_sp, SNR = cost(A)
        if (i % (N_g_itr/10) == 0) and print_costs:
            print E / N_bat, E_rec / N_bat, E_sp / N_bat, SNR

def ista(A, gradA, alpha, l):
    """
    Returns the new value of the sparse coefficients after one ista step
        for the cost function f(A) + Alpha * |A|
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
    """
    theta = alpha / l
    Ap = A - 1./l * gradA
    max = 1000000. #FIXME better function than clip?
    return np.sign(Ap) * np.clip(np.abs(Ap) - theta, 0., max)

def row_normalize(D):
    # FIXME is this right?
    norms = np.sqrt(np.diag((D**2).sum(0)))
    D = D.dot(np.diag(1./norms))
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
    """
    def __init__(self, n_dict, n_features, lamb=.1, eta=.01):
        self.n_dict = n_dict
        self.n_feature = n_features
        self.lamb = lamb
        self.eta = eta
        self.activities = None # ??
        self.reset()
        self.stale_A = True  # ??

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
        self.D = normalize(np.random.randn((n_features,
                                            n_dict)))

    def infer_A(self, X, A=None, n_steps=150, track_cost=False):
        """
        Infer sparse coefficients, A.

        Parameters
        ----------
        X : array
            A batch of input data.
        n_steps : int
            Number of steps for inference.
        """
        self.stale_A = False
        self.X = X
        cost = functools.partial(self.cost, X)
        grad = functools.partial(self.grad_A, X)
        l = calculate_l(self.D)
        A0 = np.zeros_like(A)
        A = fista(cost, grad_A, A0, n_steps, track_cost)
        if isinstance(A, tuple):
            self.A, cost = A #???
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

    def learn_D(self, X=None, A=None):
        """
        Run one step of dictionary learning on D.

        Parameters
        ----------
        X : array (optional)
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        if self.stale_A and A is None:
            raise AttributeError("Coefficients: A, are old. "+
                                 "Run infer_A before learn_D")
        if X is None:
            X = self.X
        if A is None:
            A = self.A
        self.D = normalize(D-self.eta*normalize(self.grad_D(X, A)))
        self.stale_A = True

    def train(self, data, batch_size=100, n_epochs=10, reset=True):
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
        reset : boolean (optional)
            Reset dictionary before training.
        """
        if reset:
            self.reset()
        batch_size = min(batch_size, X.shape[0])
        n_batches = math.ceil(X.shape[0]/batch_size)
        for ii in range(n_epochs):
            order = np.random.permutation(n_batches)
            for jj in order:
                batch = data[jj*batch_size:min((jj+1)*batch_size)]
                A = self.infer_A(batch)
                self.learn_D()
            print("Epoch "+str(ii)+" of "+str(n_epochs))
            print("MSE:      "+str(self.MSE(batch, A)))
            print("Sparsity: "+str(self.sparsity(batch, A)))
            print("SNR:      "+str(self.SNR(batch, A)))
            print("Cost:     "+str(self.cost(batch, A)))
            print()

    def reconstruct(self, X, A=None):
        """
        Reconstruct the data from the learned model.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        if A is None:
            A = self.infer_A(X)
        return A.dot(self.D)

    def MSE(self, X, A=None):
        """
        One-half mean-squared error for reconstructed data.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        X_prime = self.reconstruct(X, A=A)
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

    def sparsity(self, X=None, A=None):
        """
        Sparsity measure for coefficients.

        Parameters
        ----------
        X : array (optional)
            Input data.
        A : array (optional)
            Sparse coefficients.
        """
        if A is None:
            assert X is not None, 'Must provide A or X'
            A = self.infer_A(X)
        return self.lamb*np.absolute(A).mean(0).sum()

    def cost(self, X, A=None):
        """
        Sparse coding cost.

        Parameters
        ----------
        X : array
            Input data.
        A : array  (optional)
            Sparse coefficients.
        """
        return self.MSE(X, A=A)+self.sparsity(A)
