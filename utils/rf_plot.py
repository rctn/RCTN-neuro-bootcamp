import numpy as np
import matplotlib.pyplot as plt
import math

def show_fields(M, cmap = 'gray', m = None, pos_only = False):
    """
    M - dictionary with dict elements as rows (each row is an unraveled image)
    cmap - color map for plot
    m - Plot a m by m grid of receptive fields
    Plots receptive fields
    """
    N, N_pix = M.shape
    if (m == None):
        m = int(math.ceil(np.sqrt(N)))
    
    l = int(np.sqrt(N_pix)) # Linear dimension of the image
    
    mm = np.max(np.abs(M))
    
    out = np.zeros(((l + 1) * m, (l + 1) * m)) + np.max(M)

    for u in range(N):
        i = u / m
        j = u % m
        out[(i * (l + 1)):(i * (l + 1) + l), (j * (l + 1)):(j * (l + 1) + l)] = np.reshape(M[u], (l, l))
        
        
    if pos_only:
        m0 = 0
    else:
        m0 = -mm
    m1 = mm
    #plt.imshow(out, cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.imshow(out, cmap = cmap, interpolation = 'nearest',
               vmin = m0, vmax = m1)
    plt.colorbar()
