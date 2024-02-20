import numpy as np
import itertools

#########################################################################
# auc_score
# Inputs: 
    # scores: numpy list of fraud scores (higher is more likely to be fraud)
    # benign_ind: indices of benign vertices
    # fraud_ind: indices of fraud vertices
# Output: AUC score
#########################################################################

def auc_score(scores, benign_ind, fraud_ind):
    B = scores[benign_ind]
    F = scores[fraud_ind]
    return round(np.sum([np.sum(f > B) for f in F]) / (len(B)*len(F)), 4)

#########################################################################
# similarity_kendall_tau
# Inputs: 
    # A: list of ranks for each object
    # B: list of ranks for each object
    # scores: list of scores for each object where scores[x] - scores[y] gives similarity between x and y
# Output: AUC score
#########################################################################
def similarity_kendall_tau(A, B, scores):
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0

    for x, y in pairs:
        a = A[x] - A[y] # difference in ranks of x and y in A
        b = B[x] - B[y] # difference in ranks of x and y in B

        # if discordant (different signs)
        if (a * b < 0):
            distance += np.abs(scores[x] - scores[y])

    return distance
  
#########################################################################
# permute_vetices
# Inputs: 
    # A: adjacency matrix of graph (n x n matrix)
    # labels: list of labels for each vertex
# Output: new A and labels with permuted vertices
#########################################################################
def permute_vetices(A, labels):
    # Generate a random permutation of vertex labels
    perm = np.random.permutation(A.shape[0])

    # Apply the permutation to both rows and columns of the adjacency matrix
    A_new = A[np.ix_(perm, perm)]
    labels_new = labels[perm]

    return A_new, labels_new
