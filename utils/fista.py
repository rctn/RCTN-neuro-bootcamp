# Python Script that implements FISTA in theano

# Full Theano Implementation of FISTA

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def fista_updates(t_A, t_E_rec, t_Alpha, t_L, pos_only = True):
    """
    Code to generate theano variables to minimize
        t_E = t_E_rec + t_Alpha * T.sum(abs(t_A))
    with respect to t_A using FISTA

    t_A - Theano shared variable to minimize with respect to
    t_E_rec - Theano Energy function to minimize, not including abs(t_A) term
    t_L - Theano variable for the Lipschitz constant of d t_E / d t_A
    pos_only - Boolean to say if we should do positive only minimization
    
    Return - a dictionary of updates to run fista to pass to theano.function
    
    Note: The auxillary variables needed by the algorithm must be 
        reset after each run
    t_A = t_fista_X = A0, t_T = 1
    
    See the end of the script for a sample usage
    """

    def threshold(t_X):
        """
        Threshold function
        """
        return T.switch(t_X > 0., t_X, 0.)

    t_B = t_A - (1./t_L) * T.grad(t_E_rec, t_A)
    t_C = T.abs_(t_B) - t_Alpha/t_L
    t_A_ista = T.switch(t_B > 0, 1., -1.) * threshold(t_C)

    
    if pos_only:
        t_A_ista = threshold(t_A_ista)
    t_A_ista.name = 'A_ista'
    
    t_T = theano.shared(np.array([1.]).astype(theano.config.floatX), 'fista_T')
    shape = t_A.get_value().shape
    t_X = theano.shared(np.zeros(shape).astype(
            theano.config.floatX), 'fista_X')
    
    
    t_X1 = t_A_ista
    t_T1 = 0.5 * (1 + T.sqrt(1. + 4. * t_T ** 2)); t_T1.name = 'fista_T1'
    t_A1 = t_X1 + (t_T1[0] - 1)/t_T[0] * (t_X1 - t_X)
    t_A1.name = 'fista_A1'
    updates = OrderedDict()
    updates[t_A] = t_A1
    updates[t_X] = t_X1
    updates[t_T] = t_T1
    
    return updates

"""

# Sample Code Snippet to use this implementation of FISTA

A = np.zeros(...)
Alpha = ...
N_g_itr = ...

t_Alpha = T.scalar('Alpha')
t_L = T.scalar('L')

t_A = theano.shared(A, 'A')
t_E_rec = <insert energy function>(t_A, <other_params>)
t_E_sp = t_Alpha * T.sum(T.abs_(t_A))
t_E = t_E_rec + t_E_sp

fist_updates = fista_updates(t_A, t_E_rec, t_Alpha, t_L)
_, t_fista_X, t_T = fist_updates.keys()

fista_step = theano.function(inputs=[t_Alpha, t_L, <other_inputs_to_t_E_rec>],
                             outputs = [t_E, t_E_rec, t_E_sp],
                             updates = fist_updates)
def calculate_L():
   'Returns an upper bound on the Lipschitz constant of dE_rec/dA'
   pass

L = calculate_L()

# Initialize t_A, and extra variables
A0 = np.zeros_like(t_A.get_value()).astype(theano.config.floatX)
t_A.set_value(A0)
t_fista_X.set_value(A0)
t_T.set_value(np.array([1.]).astype(theano.config.floatX))

for _ in range(N_g_itr):
    X0 = np.zeros_like(t_A.get_value()).astype(theano.config.floatX)
    E, E_rec, E_sp = fista_t(Alpha, L, ...)
    print E, E_rec, E_sp


"""
