import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math 
from operator import itemgetter

import sys
sys.path.append('..') # add parent directory to path to import sbm
from sbm import gen_random_sbm, get_sbm_params
from sbm import est_sbm_params as est_sbm_params_exact
from synthetic_data_implementations.sbm_dp import estimate_sbm_params_dp
from truncate_graph import truncate_graph, add_cauchy_noise, add_laplace_noise

def run_generate_synthetic_agm(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='cauchy', n_samples=1, use_triangles=False):
    if use_triangles:
        eps = eps / 3.
    else:
        eps = eps / 2.

    # estimate parameters
    sbm_params = estimate_sbm_params_dp(A, labels, eps, deg_cutoff, fraud_private=fraud_private, noise_type=noise_type)
    degree_seq = est_degree_sequence(A, labels, eps, deg_cutoff, fraud_private=fraud_private, noise_type=noise_type)
    if use_triangles:
        # clip to at most deg_cutof^2 triangles
        n_triangles = int(np.clip(est_n_triangles(A, labels, eps, deg_cutoff, fraud_private=fraud_private, noise_type=noise_type), 0, deg_cutoff**2))
    else:
        # only estimate on fraud withot privacy
        n_triangles = int(get_n_triangles(A[labels == 1][:,labels == 1]))

    sbm_print = [round(s, 4) for s in sbm_params[2:]]
    print(f'Estimated SBM params: {sbm_print}, Estimated avg degree: {round(degree_seq.mean(), 2)}, Estimated n zero degree: {len(degree_seq[degree_seq==0])}, Estimated triangles: {n_triangles}')

    # sample graphs
    graphs = []
    for i in range(n_samples):
        if samples > 1:
            print('Sample:', i)
        graphs.append(sample_tricycle_graph_attr(sbm_params, degree_seq, n_triangles))
    
    est_params = {'sbm_params': sbm_params, 'degree_seq': degree_seq, 'n_triangles': n_triangles}
    return graphs, est_params

def neighbors(A, i):
    return A[:, [i]].nonzero()[1]

def acceptance_prob(i,j,X,A):
    if A is None:
        return 1
    if X[i] == 0 and X[j] == 0:
        return A[0]
    elif X[i] == 1 and X[j] == 1:
        return A[1]
    else:
        return A[2]

def sample_tricycle_graph_attr(sbm_params, degree_seq, n_tri, iters=1000):
    _, n1, p0, p1, p01 = sbm_params
    n0 = len(degree_seq) - n1

    labels = np.random.permutation(np.hstack((np.zeros(n0), np.ones(n1)))) # attributes of nodes
    E = sample_tricycle_graph(degree_seq, n_tri) # sample edges
    _, _, p0_samp , p1_samp, p01_samp = est_sbm_params_exact(E, labels)
    # acceptance probabilities for edges between attributes
    A_old = None 

    for i in range(iters):
        # print('Iteration:', i)

        R = np.array([p0 / p0_samp, p1 / p1_samp, p01 / p01_samp])

        if A_old is not None:
            # elemnt-wise multiplication of A and R
            R = A * R
        
        A = R / max(R)

        if A_old is not None and (A - A_old).sum() <= 1e-6:
            # print('Probabilities converged to within 1e-6.')
            break
        
        A_old = A

        print(f'Graph Sample Iteration {i}, Acceptance probabilities: {A}')

        E = sample_tricycle_graph(degree_seq, n_tri, X=labels, A=A) # re-sample edges
    
    return E, labels

def sample_tricycle_graph(degree_seq, n_tri, X=None, A=None):
    v_list = np.where(degree_seq > 1)[0]
    pi = degree_seq[v_list] / np.sum(degree_seq[v_list]) # distribution of nodes to sample from 
    G = expected_degree_graph(degree_seq, selfloops=False, X=X, A=A) 
    edges = G.edges() # all edges sorted from oldest to newest    

    # get adjacency matrix
    E_T = nx.to_scipy_sparse_array(G)
    tau = get_n_triangles(E_T)

    max_iter = len(degree_seq) 
    iter = 0
    # sample edges until we have n_tri triangles
    while tau < n_tri and iter < max_iter:
        if iter % 1000 == 0:
            print(f'Triangle-Adding Iteration {iter}, Number of triangles: {tau}')
        i = np.random.choice(v_list, p=pi) 
        N_i = neighbors(E_T, i)

        iter += 1
        
        if len(N_i) == 0:
            continue
        j = np.random.choice(N_i)

        N_j = neighbors(E_T, j)
        if len(N_j) == 0:
            continue

        k = np.random.choice(neighbors(E_T, j))
        if k == i:
            continue
           
        if E_T[i,j] != 1 and np.random.rand() <= acceptance_prob(i, j, X, A): # if edge (i,j) not in graph
            q,r = edges[0] # get oldest edge to replace
            CN_qr = neighbors(q).dot(neighbors(r)) # get number of common neighbors of q and r
            CN_ij = neighbors(i).dot(neighbors(j)) # get number of common neighbors of i and k

            if CN_qr < CN_ij: # adding edge ij would increase num triangles
                E_T[q,r] = 0
                E_T[r,q] = 0
                E_T[i,j] = 1
                E_T[j,i] = 1
                edges = edges[1:] + [(i,j)]
                tau += CN_ij - CN_qr
            else:
                edges = edges[1:] + [(q,r)]    
    return E_T 


################################################################################################
##### Estimate number of triangles and degree sequence of graph with differential privacy ######
################################################################################################

def get_n_triangles(A):
    return A.power(3).trace() / 6

def est_n_triangles(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='laplace', delta=1e-8):
    # add noise to degree sequence proportional to D / eps to guarantee DP
    A_trunc, labels_trunc = truncate_graph(A, labels, deg_cutoff, fraud_private)
    n_triangles = get_n_triangles(A)

    # from https://arxiv.org/pdf/1208.4586.pdf
    sens = 3*deg_cutoff**2 
    if noise_type == 'laplace':
        n_triangles += add_laplace_noise(eps, delta, sens, A_trunc, labels_trunc, deg_cutoff,
                                    n_samples=1, truncate_fraud=fraud_private)
    if noise_type == 'cauchy':
        n_triangles += add_cauchy_noise(eps, sens, A_trunc, labels_trunc, deg_cutoff,
                                    n_samples=1, truncate_fraud=fraud_private)
    return np.max(n_triangles, 0)

def est_degree_sequence(A, labels, eps, deg_cutoff, fraud_private=False, noise_type='laplace', delta=1e-8):
    # add noise to degree sequence proportional to D / eps to guarantee DP
    A_trunc, labels_trunc = truncate_graph(A, labels, deg_cutoff, fraud_private)

    deg_list = sorted(np.sum(A_trunc, axis=0).tolist()[0])

    sens = deg_cutoff
    if noise_type == 'laplace':
        deg_list += add_laplace_noise(eps, sens, A_trunc, labels_trunc, deg_cutoff,
                                    n_samples= len(deg_list), truncate_fraud=fraud_private)
    else:
        deg_list += add_cauchy_noise(eps, sens, A_trunc, labels_trunc, deg_cutoff,
                                    n_samples= len(deg_list), truncate_fraud=fraud_private)
    
    deg_list = post_process_deg(deg_list)
    # clip all nodes to have degree at least 1
    deg_list = np.round(np.clip(deg_list, 0, deg_cutoff)).astype(int)

    return deg_list
    
# post-processing of sorted list from 
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5360242
def post_process_deg(deg_list):
    def M(i,j):
        if i > j:
            return 0
        if i == len(deg_list):
            return 0
        if i == j:
            return deg_list[i]
        return np.mean(deg_list[i:j+1])
    
    J = []
    J.append(len(deg_list))
    for k in range(len(deg_list)-1, 0, -1):
        j_star = k
        j = J[-1]
        while len(J) > 0 and M(j_star + 1, j) <= M(k, j_star):
            j_star = j
            J.pop()
            if len(J) > 0:
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

# Returns a random graph with given expected degrees.
def expected_degree_graph(w, selfloops=False, X=None, A=None):  
    n = len(w)
    m = sum(w) / 2
    G = nx.empty_graph(n)

    # If there are no nodes are no edges in the graph, return the empty graph.
    if n == 0 or max(w) == 0:
        return G

    rho = 1 / sum(w)
    # Sort the weights in decreasing order. The original order of the
    # weights dictates the order of the (integer) node labels, so we
    # need to remember the permutation applied in the sorting.
    order = sorted(enumerate(w), key=itemgetter(1), reverse=True)
    mapping = {c: u for c, (u, v) in enumerate(order)}
    seq = [v for u, v in order]
    last = n
    if not selfloops:
        last -= 1
    
    while G.number_of_edges() < m:
        for u in range(last):
            v = u
            if not selfloops:
                v += 1
            factor = seq[u] * rho
            p = min(seq[v] * factor, 1)
            while v < n and p > 0:
                if p != 1:
                    r = np.random.rand()
                    v += math.floor(math.log(r, 1 - p))
                if v < n:
                    q = min(seq[v] * factor, 1)
                    if np.random.rand() < q / p and np.random.rand() <= acceptance_prob(mapping[u],mapping[v],X,A):
                        G.add_edge(mapping[u], mapping[v])
                    v += 1
                    p = q
    return G

if __name__ == '__main__':
    # Example usage
    n = 8000
    params = get_sbm_params(n, 10, 20, 3, 0.1)
    print(params)
    A, labels = gen_random_sbm(params)

    n_triangles = get_n_triangles(A)
    print(f'Triangles: {n_triangles}')
    deg_list = np.sum(A, axis=0).tolist()[0]
    sbm_params = est_sbm_params_exact(A, labels)

    print(sbm_params)

    samples, params = run_generate_synthetic_agm(A, labels, 100., 25, fraud_private=False, noise_type='cauchy', n_samples=1, use_triangles=True)

    for sample in samples:
        A, labels = sample
        print(est_sbm_params_exact(A, labels))
    
    print (f'Estimated SBM params: {params["sbm_params"]}, Estimated avg degree: {round(params["degree_seq"].mean(), 2)}, Estimated n zero degree: {len(params["degree_seq"][params["degree_seq"]==0])}, Estimated triangles: {params["n_triangles"]}')
    