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

# take mean of AUCs and add Laplace noise 
def agg_mean_laplace(aucs, eps):
    k = len(aucs)
    b = 1. / (k * eps)
    return np.clip(np.mean(aucs) +  np.random.laplace(0, b, 1), 0, 1)[0]

# take median of AUCs using inverse sensitivity mechanism
def agg_median_inverse_sensitivity(aucs, eps):
    aucs = sorted(aucs + np.random.rand(len(aucs))/1000) # add a tiny bit of random noise to break ties

    if np.isnan(aucs[0]) and np.isnan(aucs[-1]):
        return np.nan

    n = len(aucs)
    # median of AUCs
    m = np.percentile(aucs, 50)
    if m in aucs:
        m_ind_min = np.where(aucs == m)[0].min()
        m_ind_max = np.where(aucs == m)[0].max()
    else:
        m_ind_min = int(n/2)
        m_ind_max = int((n/2) + 1)

    dists = {}
    s = 0
    for i,auc in enumerate(aucs):
        if auc < m:
            dists[(s, auc)] = m_ind_min - i
        elif auc > m:
            dists[(s, auc)] = i - m_ind_max
        else:
            if i == m_ind_min:
                dists[(s, auc)] = int(n/2) - m_ind_min + 1
        s = auc
    if s == m:
        dists[(s, 1)] = len(aucs) - int(n/2)
    else: 
        dists[(s, 1)] = len(aucs) - m_ind_max

    sample_weights = [np.exp(-eps * dist / 2) for dist in dists.values()]
    sample_interval = list(dists.keys())[np.random.choice(range(len(dists.keys())), p=sample_weights/sum(sample_weights))]
    return np.random.uniform(sample_interval[0], sample_interval[1])
 
# if __name__ == '__main__':
#     aucs = np.array([0.8, 0.9, 0.85, 0.7, 0.95])
#     print(agg_mean_laplace(aucs, 1.0))
#     print(agg_median_inverse_sensitivity(aucs, 1.0))
   