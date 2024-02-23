### IMPLEMENTATIONS OF FRAUD DETECTORS TO BENCHMARK ###

import numpy as np
import scipy
import dsd 
import networkx as nx
import igraph as ig 
from helpers import to_igraph


def score_random(A):
    return np.random.rand(A.shape[0])

####################################################################################################
############################## Rank by degree of vertex in graph ##################################
####################################################################################################
def score_by_degree(A, negate=False):
    return np.asarray(np.sum(A, axis=1) * (-1 if negate else 1)).ravel()

####################################################################################################
############################## Rank by # clustering coefficient of each vertex in graph ############
####################################################################################################
def score_by_clustering_coeff(A, negate=False, igraph=False):
    if igraph is False:
        g = to_igraph(A)
    else:
        g = A
    return np.array(g.transitivity_local_undirected()) * (-1 if negate else 1)

####################################################################################################
############################## Community detection score ###########################################
####################################################################################################
def score_by_community_detection(A, igraph=False):
    if igraph is False:
        g = to_igraph(A)
    else:
        g = A
    c = g.community_leiden() 
    # score by 1 / size of cluster 
    scores = [1. / len(c[m]) for m in c.membership]
    return np.array(scores)
   

####################################################################################################
############ Rank by reconstruction error from low rank approximation ##############################
####################################################################################################

# approximate matrix "S" with matrix of rank "rank"
def threshold(S, rank, sparse=False):
    if sparse:  
        U, s, Vh = scipy.sparse.linalg.svds(S, k=rank)
    else:
        U, s, Vh = scipy.linalg.svd(S, full_matrices=False)
    k = s.size
    if k < rank:
        return S
    for i in range(rank, k):
        s[i] = 0
    return U @ np.diag(s) @ Vh


# get element-wise reconstruction errors when approximating matrix "M" with rank "rank"
def get_error(M, rank, sparse):
    assert(rank != None)
    # get low rank approximation
    L = threshold(M, rank, sparse=sparse)
    delta = []
    delta = np.abs(L - M)
    return delta

# aggregate the reconstruction error per row
def agg_error(delta, method='sum'):
    if method == 'sum':
        diffs = list(np.sum(delta, axis=1))
    elif method == 'max':
        diffs = list(np.amax(delta, axis=1))
    elif method == 'L2':
        diffs = list(np.sqrt(np.sum(delta**2, axis=1)))
    else:
        assert(False)
    return np.array(diffs).reshape(-1)

# take in matrix of dim n x d, return n scores
# algorithm approximates matrix with rank "rank" and scores based 
# on reconstruction error per-row
def score_by_truncate_svd(A, rank, agg_method='sum', sparse=True):
    try:
        return agg_error(get_error(A, rank, sparse), agg_method)
    except:
        return np.ones(np.shape(A)[0])
    
####################################################################################################
######################################## Dense Subgraphs ###########################################
####################################################################################################
    
def score_dense_subgraph(A):
    g = nx.from_scipy_sparse_array(A)
    inds = dsd.greedy_charikar(g)   
    scores = np.zeros(A.shape[0])
    scores[inds] = 1
    return scores 
    
####################################
######## Combination of Scores #####
####################################

def replace_zero(m):
    if m == 0:
        return 1
    return m

def agg_scores(score_list, weights=None, agg_func=np.sum):
    # give equal weight to all scores by default
    if weights is None:
        weights = np.ones(len(score_list)) / len(score_list)
    norm_scores = [weights[i] * (scores / replace_zero(max(scores))) for i, scores in enumerate(score_list)]
    return agg_func(norm_scores, axis=0)


