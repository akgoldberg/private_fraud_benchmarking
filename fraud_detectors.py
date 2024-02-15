### IMPLEMENTATIONS OF FRAUD DETECTORS TO BENCHMARK ###

import numpy as np
import scipy
import fraud_detector_implementations.telltail as telltail
import heapq

####################################################################################################
############################## Rank by degree of vertex in graph ##################################
####################################################################################################
def score_by_degree(A, negate=False):
    return np.asarray(np.sum(A, axis=1) * (-1 if negate else 1)).ravel()

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
    return agg_error(get_error(A, rank, sparse), agg_method)


####################################################################################################
### Fraud Detection Algorithms Existing Implementations ###########################################
####################################################################################################

# TellTail
def score_by_telltail(A):
    # rng_init is 0
    fraud_nodes = telltail.run_telltail(A.toarray(),   telltail.get_local_search_init(A, fast=True), np.random.RandomState(0))
    scores = np.zeros(A.shape[0])
    scores[fraud_nodes] = 1
    return scores

####################################
######## Combination of Scores #####
####################################

def replace_zero(m):
    if m == 0:
        return 1
    return m

def agg_scores(score_list, weights=None):
    # give equal weight to all scores by default
    if weights is None:
        weights = np.ones(len(score_list)) / len(score_list)
    return np.sum([weights[i] * scores / replace_zero(max(scores)) for i, scores in enumerate(score_list)], axis=0)

####################################################################################################
############## Rank edges by shortest path length between its endpoints ############################
####################################################################################################

# https://fc16.ifca.ai/preproceedings/02_Molloy.pdf

def get_neighbors(A, v):
    return np.nonzero(A[v])[1]

def dijkstra_exclude_edge(A, start, exclude_edge):
    # Initialize distances and visited set
    distances = np.inf * np.ones(A.shape[0])
    distances[start] = 0
    visited = set()

    # Priority queue to store (distance, node) pairs
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Skip if already visited
        if current_node in visited:
            continue

        # Mark the current node as visited
        visited.add(current_node)

        # Update distances for neighboring nodes, excluding the specified edge
        for neighbor in get_neighbors(A, current_node):
            if (current_node, neighbor) == exclude_edge or (neighbor, current_node) == exclude_edge:
                continue  # Skip the specified edge

            distance = current_distance + 1
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def shortest_paths_for_each_edge(A):
    all_shortest_paths = {}

    nodes = np.arange(A.shape[0])

    for node in nodes:
        all_shortest_paths[node] = {}

    for node in nodes:
        for neighbor in get_neighbors(A, node):
            edge = (node, neighbor)
            reverse_edge = (neighbor, node)

            if edge not in all_shortest_paths[node] and reverse_edge not in all_shortest_paths[node]:
                shortest_distances = dijkstra_exclude_edge(A, node, edge)
                all_shortest_paths[node][edge] = shortest_distances

    return all_shortest_paths

###################################################
############## OddBall ############################
###################################################


###################################################
################# CADA ############################
###################################################