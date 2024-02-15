import numpy as np
import scipy

### Stochastic Block Model (SBM) functions ###

# Create an upper triangular matrix with values in the upper triangle (above diagonal)
def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)

## A contains n0 benign followed by n1 fraud
# Generate (n0 + n1) x (n0 + n1) edge adjacency matrix
def gen_random_sbm(sbm_params, sparse=True):
    n0, n1, p0, p1, p01 = sbm_params
    A00 = create_upper_matrix(np.random.binomial(1, p0, int(n0*(n0-1)/2)), n0)
    A11 =  create_upper_matrix(np.random.binomial(1, p1, int(n1*(n1-1)/2)), n1)
    A01 = np.random.binomial(1, p01, (n0,n1))
    
    A = np.zeros((n0+n1, n0+n1))
    A[n0:, n0:] = A11
    A[0:n0, 0:n0+n1] = np.hstack((A00, A01))

    if sparse:
        A_out = scipy.sparse.csr_matrix(A + A.T)
    else:
        A_out = A + A.T
    labels = np.zeros(n0+n1)
    labels[n0+1:] = 1
    
    return A_out, labels

# Get sbm params fixing rate of fraud r, expected degree of fraud nodes at d0 + d01 and benign at d0 + r*d01
# r is rate of ratio of fraud to benign nodes, d0 is expected # of edges benign<->benign, d1 is expected # of edges fraud<->fraud, d01 is expected # of edges benign->fraud
def get_sbm_params(n0, d0, d1, d01, r):
    n1 = int(n0*r)
    p0 = d0 / (n0-1)
    p1 = d1 / (n1-1)
    p01 = d01 / n0
    return (n0, n1, p0, p1, p01)

# Go from sbm params to  expected degrees of fraud and benign nodes
def from_sbm_params(sbm_params):
    n0, n1, p0, p1, p01 = sbm_params
    r = 1. * n1 / n0
    d0 = p0 * (n0-1)
    d1 = p1 * (n1-1)
    d01 = p01 * n0
    return n0, d0, d1, d01, r
