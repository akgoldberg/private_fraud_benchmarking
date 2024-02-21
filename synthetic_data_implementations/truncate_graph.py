import numpy as np
import scipy 

# truncate the graph by removing vertices with degree greater than max_deg
# Inputs: 
    # A: adjacency matrix of graph (n x n matrix)
    # D: maximum degree allowed for vertices
# Output: truncated A and labels
def truncate_graph(A, labels, D, truncate_fruad=False):
    deg = np.array(np.sum(A, axis=0).tolist()[0])
    if truncate_fruad:
        # only keep vertices with degree <= max_deg
        ind = np.where(deg <= D)[0]
    else:
        # only keep vertices with degree <= max_deg or are fraud
        ind = np.where((deg <= D) | (labels == 1))[0]
    return A[ind][:,ind], labels[ind]

# get the beta-smooth sensitivity of the graph truncation mechanism
def get_smooth_sensitivity(A, labels, D, beta, truncate_fruad=False):
    deg = np.sum(A, axis=1)
    if not truncate_fruad:
        deg = deg[np.where(labels != 1)[0]] # only consider benign vertices as contributers to sensitivity
    return max([np.exp(-beta * k)*len(deg[(deg >= D - k) & (deg <= D + k + 1)]) for k in range(A.shape[0] - D - 2)])

def get_noise_magnitude(eps, delta, sens, A, labels, D, noise_type, truncate_fraud):
    if noise_type == 'laplace':
         beta = -eps / (2*np.log(2*delta))
    if noise_type == 'cauchy':
          beta = eps / np.sqrt(2)
    
    print('Beta:', beta)
    
    S_truncate = get_smooth_sensitivity(A, labels, D, beta, truncate_fraud)
    S = S_truncate * sens # smooth sensitivity
    if noise_type == 'laplace':
        return 2*S/eps
    if noise_type == 'cauchy':
        return S * np.sqrt(2) / eps

# given (D-restricted sensitivity) of a mechanism, get Laplace noise draws to add after truncation to guarantee eps, delta DP
def add_laplace_noise(eps, delta, sens, A, labels, D, n_samples=1, truncate_fraud=False):
    S = get_noise_magnitude(eps, delta, sens, A, labels, D, 'laplace', truncate_fraud)
    print(f'Eps: {eps}, Noise magnitude: {2*S/eps}')
    if n_samples > 1:
        return np.random.laplace(0, S/eps, n_samples)
    return np.random.laplace(0, S/eps, 1)[0]

# given (D-restricted sensitivity) of a mechanism, get Cauchy noise draws to add after truncation to guarantee eps DP
def add_cauchy_noise(eps, sens, A, labels, D, n_samples=1, truncate_fraud=False):
    S = get_noise_magnitude(eps, 0, sens, A, labels, D, 'cauchy', truncate_fraud) # smooth sensitivity
    print(f'Eps: {eps}, Noise magnitude: {S * np.sqrt(2) / eps}')
    if n_samples > 1:
        return scipy.stats.cauchy(0, S/eps).rvs(n_samples)
    return scipy.stats.cauchy(0, S/eps).rvs(1)[0]