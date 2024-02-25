from truncate_graph import truncate_graph, add_cauchy_noise, add_laplace_noise
import numpy as np
import sys 

sys.path.append('..') # add parent directory to path to import sbm
from sbm import gen_random_sbm, get_sbm_params, est_sbm_params

def run_generate_synthetic_sbm(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='cauchy', n_samples=1, non_private=False, stats_only=False):
    
    sbm_params_exact = est_sbm_params(A, labels)
    if non_private:
        sbm_params = sbm_params_exact
    else:
        sbm_params = estimate_sbm_params_dp(A, labels, eps, deg_cutoff, fraud_private=fraud_private, noise_type=noise_type)

    if stats_only:
        est_params = {'sbm_params': sbm_params}
        true_params = {'sbm_params': sbm_params_exact}
        return [], est_params, true_params

    # sample graphs
    graphs = []
    for i in range(n_samples):
        if n_samples > 1:
            print('Sample:', i)
        graphs.append(gen_random_sbm(sbm_params))
    
    return graphs, {'sbm_params': sbm_params}, {'sbm_params': sbm_params_exact}


def estimate_fraud_count(labels, eps):
    n0 = len(np.where(labels != 1)[0])
    n1 = len(np.where(labels == 1)[0])

    # add noise to benign count
    n0 = int(n0 + np.random.laplace(0, 1/eps, 1)[0])
    return n0, n1

def estimate_sbm_params_dp(A, labels, eps, deg_cutoff, fraud_private = True, noise_type = 'cauchy', delta=1e-8):
    # estimate the parameters of an SBM from a given graph
    # Inputs: 
        # A: adjacency matrix of graph (n x n matrix)
        # labels: list of labels for each vertex
    # Output: estimated parameters of SBM
    
    # estimate num fraud and benign nodes
    eps_count = eps / 10.
    n0, n1 = estimate_fraud_count(labels, eps_count)

    # print(f'Estimated benign count: {n0}, Estimated fraud count: {n1}')
    
    # estimate probabilities within and between fraud and benign nodes
    eps_probs = eps - eps_count
    # truncate the graph
    trunc_A, trunc_labels = truncate_graph(A, labels, deg_cutoff, fraud_private)

    # print(f'Truncated {len(labels) - len(trunc_labels)} vertices')

    ind0 = np.where(trunc_labels != 1)[0]
    ind1 = np.where(trunc_labels == 1)[0]
    # estimate counts of edges between fraud and benign nodes
    n00 = np.sum(trunc_A[ind0][:,ind0]) / 2
    n11 = np.sum(trunc_A[ind1][:,ind1]) / 2
    n01 = np.sum(trunc_A[ind0][:,ind1]) 

    # print(f'True n00: {n00}, True n11: {n11}, True n01: {n01}')

    sens = deg_cutoff
    if fraud_private:
        # weight eps by size of denominator
        eps_weights = np.array([n0*(n0-1)/2, n1*(n1-1)/2, n0*n1])
        eps_weights = 1. / np.power(eps_weights, 1./3)
        eps0, eps1, eps2 = eps_weights / np.sum(eps_weights) * eps_probs

        if noise_type == 'laplace':
            n00 += add_laplace_noise(eps0, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n11 += add_laplace_noise(eps1, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n01 += add_laplace_noise(eps2, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
        else: 
            n00 += add_cauchy_noise(eps0, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n11 += add_cauchy_noise(eps1, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n01 += add_cauchy_noise(eps2, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
    else:
        # weight eps by size of denominator
        eps_weights = np.array([n0*(n0-1)/2, n0*n1])
        eps_weights = 1. / np.power(eps_weights, 1.)

        eps0, eps1 = eps_weights / np.sum(eps_weights) * eps_probs


        if noise_type == 'laplace':
            n00 += add_laplace_noise(eps0, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n01 += add_laplace_noise(eps1, delta, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
        else:   
            n00 += add_cauchy_noise(eps0, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
            n01 += add_cauchy_noise(eps1, sens, A, labels, deg_cutoff, truncate_fraud=fraud_private)
    
    # print(f'Estimated n00: {n00}, Estimated n11: {n11}, Estimated n01: {n01}')

    p0 = n00 / (n0 * (n0 - 1) / 2)
    p1 = n11 / (n1 * (n1 - 1) / 2)
    p01 = n01 / (n0 * n1)

    # clip lower to lower bounds of at least 0.1 expected edge between each group
    lower_p0 = 0.1 / n0
    lower_p1 = 0.1 / n1 
    lower_p01 = 0.1 / n0

    p0 = np.clip(p0, lower_p0, 0.8)
    # p1 = np.clip(p1, lower_p1, 0.8) # dont clip since no noise added
    p01 = np.clip(p01, lower_p01, 0.8)

    print(f'Estimated p0: {p0}, Estimated p1: {p1}, Estimated p01: {p01}')

    return (n0, n1, p0, p1, p01)

if __name__ == "__main__":
    # Example usage
    n = 1000
    params = get_sbm_params(n, 5, 10, 3, 0.1)
    print(params)
    A, labels = gen_random_sbm(params)
    eps = 1.
    deg_cutoff = 25
    estimate_sbm_params_dp(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='laplace')
