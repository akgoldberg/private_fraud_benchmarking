#########################################################################
############## IMPLEMENTATION OF SUBSAMPLE AND AGGREGATE ALGO ###########
#########################################################################

import numpy as np

#########################################################################
# partition_duplicate_graph
# Inputs: 
    # A: adjacency matrix of graph (n x n matrix)
    # benign_ind: indices of benign vertices
    # fraud_ind: indices of fraud vertices
    # k: number of partitions to split the benign vertices into
    # sub_rate: fraction of fraud vertices to subsample per partition

# Output: list of k subgraphs with benign vertices before fraud vertices
#########################################################################

def partition_duplicate_graph(A, benign_ind, fraud_ind, k, sub_rate, random_seed=0):
    np.random.seed(random_seed)

    n0 = len(benign_ind)
    sz = int(n0 / k) # number of benign vertices per partition
    sub_sz = int(len(fraud_ind) * sub_rate) # number of fraud vertices to subsample per partition 
    starts = np.arange(k) * (int(n0 / k)) # start indices of benign vertices

    # helper: get a single partition of A with subsampled fraud vertices
    def get_partition_subsamp(A, s, sz, sub_sz, benign_ind, fraud_ind):
        # choose indices of subsampled fraud vertices
        sub_ind_fraud = np.random.choice(fraud_ind, size=sub_sz, replace=False)
        sub_ind_benign = benign_ind[s:s+sz]
        sub_ind = np.hstack((sub_ind_benign, sub_ind_fraud)) # benign come before fraud

        labels = np.zeros(len(sub_ind_benign) + len(sub_ind_fraud))
        labels[len(sub_ind_benign):] = 1

        return A[sub_ind,:][:, sub_ind], labels
    
    subgraphs = [get_partition_subsamp(A,s,sz,sub_sz,benign_ind,fraud_ind) for s in starts]    
    
    return subgraphs

