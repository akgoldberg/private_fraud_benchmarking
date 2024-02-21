from truncate_graph import truncate_graph, add_cauchy_noise, add_laplace_noise, get_noise_magnitude
import numpy as np 

import sys
sys.path.append('..')
from sbm import gen_random_sbm, get_sbm_params

def run_topmfilter(A, labels, eps, deg_cutoff, n_samples=1, fraud_private=False, delta=1e-6):
    A_trunc, labels_trunc = truncate_graph(A, labels, deg_cutoff, fraud_private)

    # estimate number of edges 
    eps_count = eps / 10.
    eps_adj = eps - eps_count

    sens = deg_cutoff
    m =  int(A_trunc.nnz/2. + add_laplace_noise(eps_count, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private))

    # find threshold
    S = get_noise_magnitude(eps_adj, delta, sens, A, labels, deg_cutoff, 'laplace', fraud_private)
    eps_t = np.log(n*(n-1)/(2*m) - 1)
    eps_actual = eps_adj / S
    print(f'Eps actual: {eps_actual}')

    if eps_actual < eps_t:
        theta = (1 / (2*eps_actual)) * np.log(n*(n-1)/(2*m) - 1)
    else:
        theta = (1/eps_actual) * (np.log(n*(n-1)/(4*m) + 0.5*(np.exp(eps_actual) -1)))
    
    print('Threshold:', theta)

    A_out = np.zeros(A_trunc.shape)

    # process 1-cells (edges)
    n1 = 0
    row_ind, col_ind = A_trunc.nonzero()
    ind_list = [(i,j) for (i,j) in zip(row_ind, col_ind) if i < j]
    row_ind, col_ind = zip(*ind_list)
    one_cells = A_trunc[row_ind,col_ind]
    print(one_cells.shape)

    one_cells += add_laplace_noise(eps_adj, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private, n_samples=one_cells.shape[1])
    one_cells = (one_cells > theta).astype(int)
    n1 = one_cells.sum()
    print('n1:', n1)
    print('Processing 1-cells')
    # for i, j in inds:
    #     if i > j:
    #         continue

    #     if A_trunc[i,j] + add_laplace_noise(eps_count, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private) > theta:
    #         A_out[i,j] = 1
    #         A_out[j,i] = 1
    #         n1 += 1
    
    # process 0-cells
    print('Processing 0-cells')
    n0 = m  - n1
    print(f'Adding {n0} edges')
    while n0 > 0:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue

        A_out[i,j] = 1
        A_out[j,i] = 1
        n0 -= 1
    
    return A_out, labels_trunc, {'n_edges': m}


if __name__ == '__main__':
    n = 1000
    params = get_sbm_params(n, 5, 10, 3, 0.1)
    print(params)
    A, labels = gen_random_sbm(params)
    eps = 10.
    deg_cutoff = 25
    
    A_out, labels_out, params = run_topmfilter(A, labels, eps, deg_cutoff, fraud_private=False, delta=1e-6)
