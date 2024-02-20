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

# given (D-restricted sensitivity) of a mechanism, get Laplace noise draws to add after truncation to guarantee eps, delta DP
def add_laplace_noise(eps, delta, sens, A, labels, D, n_samples=1, truncate_fraud=False):
    beta = -eps / (2*np.log(delta))
    S_truncate = get_smooth_sensitivity(A, labels, D, beta, truncate_fraud)
    S = S_truncate * sens # smooth sensitivity
    print(f'Eps: {eps}, Noise magnitude: {2*S/eps}')
    if n_samples > 1:
        return np.random.laplace(0, 2*S/eps, n_samples)
    return np.random.laplace(0, 2*S/eps, 1)[0]

# given (D-restricted sensitivity) of a mechanism, get Cauchy noise draws to add after truncation to guarantee eps DP
def add_cauchy_noise(eps, sens, A, labels, D, n_samples=1, truncate_fraud=False):
    beta = eps / np.sqrt(2)
    S_truncate = get_smooth_sensitivity(A, labels, D, beta, truncate_fraud)
    S = S_truncate * sens # smooth sensitivity
    print(f'Eps: {eps}, Noise magnitude: {S * np.sqrt(2) / eps}')
    if n_samples > 1:
        return scipy.stats.cauchy(0, S*np.sqrt(2) / eps).rvs(n_samples)
    return scipy.stats.cauchy(0, S*np.sqrt(2) / eps).rvs(1)[0]