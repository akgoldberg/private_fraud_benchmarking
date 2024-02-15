import numpy as np

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
   
