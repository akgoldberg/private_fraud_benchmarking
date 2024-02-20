import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('..') # add parent directory to path to import sbm
from sbm import gen_random_sbm, get_sbm_params

from estimate_sbm_params_dp import estimate_sbm_params_dp
from truncate_graph import truncate_graph, add_cauchy_noise, add_laplace_noise

def est_n_triangles():
    pass

def est_degree_sequence(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='laplace', delta=1e-8):
    # add noise to degree sequence proportional to D / eps to guarantee DP
    A_trunc, _ = truncate_graph(A, labels, deg_cutoff, fraud_private)

    deg_list = sorted(np.sum(A_trunc, axis=0).tolist()[0])

    sens = deg_cutoff
    if noise_type == 'laplace':
        deg_list += add_laplace_noise(eps, delta, sens, A, labels, deg_cutoff,
                                    n_samples= len(deg_list), truncate_fraud=fraud_private)
    else:
        deg_list += add_cauchy_noise(eps, sens, A, labels, deg_cutoff,
                                    n_samples= len(deg_list), truncate_fraud=fraud_private)
    
    print(deg_list[:10])

    deg_list = post_process_deg(deg_list)
    deg_list = np.round(np.clip(deg_list, 0, deg_cutoff)).astype(int)

    deg_seq = np.bincount(deg_list)

    return deg_seq
    
# post-processing of sorted list from 
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5360242
def post_process_deg(deg_list):
    def M(i,j):
        return np.mean(deg_list[i:j+1])
    
    J = []
    J.append(len(deg_list))
    for k in range(len(deg_list), 1, -1):
        j_star = k
        j = J[-1]
        while J and M(j_star + 1, j) <= M(k, j_star):
            j_star = j
            J.pop()
            j = J[-1]
        J.append(j_star)
    b = 1
    s = np.zeros(len(deg_list))
    while J:
        j_star = J.pop()
        for k in range(b, j_star):
            s[k] = M(b, j_star)
        b = j_star + 1
    return s
        
if __name__ == '__main__':
    n = 1000
    params = get_sbm_params(n, 5, 10, 3, 0.1)
    print(params)
    A, labels = gen_random_sbm(params)
    eps = 5.
    deg_cutoff = 25
    deg_seq = np.bincount(np.sum(A, axis=0).tolist()[0])
    deg_seq_noisy = est_degree_sequence(A, labels, eps, deg_cutoff, 
                                  fraud_private=False, noise_type='cauchy', delta=1e-8)
    
    plt.hist(deg_seq)
    plt.hist(deg_seq_noisy)
    print(sum(deg_seq), sum(deg_seq_noisy))
    plt.show()