def calculate_L():
    """
    Calculates the 'L' constant for FISTA for the dictionary in t_D.get_value()
    """
    D = t_D.get_value()
    try:
        L = 2 * eigh(np.dot(D, D.T), eigvals_only=True, eigvals=(N_sp-1,N_sp-1))[0]
    except ValueError:
        L = (2 * std ** 2 * N_sp).astype('float32') # Upper bound on largest eigenvalue
    # std - standard deviation of the dictionary elements
    return L
    
 
t_B = t_A - (1./t_L) * t_gEA
t_C = T.abs_(t_B) - t_Alpha/t_L
t_Ap = T.switch(t_B > 0, 1., -1.) * T.switch(t_C > 0, t_C, 0.)
t_Ap.name = 'Ap'

ista = theano.function(inputs = [t_I_idx, t_Alpha, t_L],
                       outputs = [t_Ap],
                       updates = [(t_A, threshold(t_Ap))])

def fista(I_idx, Alpha, L, N_g_itr, X0, print_costs = False):
    """
    Computes FISTA Approximation
    """
    Xs = np.zeros((2, N_bat, N_sp)).astype('float32')
    Ys = np.zeros((2, N_bat, N_sp)).astype('float32')
    Ts = np.zeros((2,)).astype('float32')
    #np.zeros((N_bat, N_sp)).astype('float32')
    Xs[0] = X0
    Ys[0] = X0
    Ts[0] = 1.
    for i in range(N_g_itr-1):
        i0 = i % 2
        i1 = i % 2
        t_A.set_value(Ys[i0])
        E, E_rec, E_sp, SNR = costs(I_idx, Alpha)
        if (i % (N_g_itr/10) == 0) and print_costs:
            print E / N_bat, E_rec / N_bat, E_sp / N_bat, SNR
        Xs[i1] = ista(I_idx, Alpha, L)[0]
#        coeff_thresh()
        Ts[i1] = 0.5 * (1. + np.sqrt(1. + 4. * Ts[i0] ** 2))
        Ys[i1] = Xs[i1] + (Ts[i0]-1)/ Ts[i1] * (Xs[i1]-Xs[i0])
    
