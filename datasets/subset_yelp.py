import scipy 
import numpy as np
import sys
sys.path.append('..') # add parent directory to path to import partition_aggregate
from partition_aggregate import partition_duplicate_graph

def subset_yelp(divide_rate=4):
    PATH = 'yelp_dataset'
    kind = 'homo'
    mat = scipy.io.loadmat(f'{PATH}/YelpChi.mat')
    A = mat[kind]   
    labels = mat['label'].reshape(-1) # 0 is benign, 1 is fraud, -1 is unknown
    metadata = mat['features']
    benign_ind = np.where(labels != 1)[0]
    fraud_ind = np.where(labels == 1)[0]
    # subsample one quarter of the graph (equal proportions benign and fraud)
    A_sub, label_sub = partition_duplicate_graph(A, benign_ind, fraud_ind, divide_rate, 1. / divide_rate)[0]

    scipy.sparse.save_npz(f'{PATH}/A_sub.npz', A_sub)
    np.savetxt((f'{PATH}/labels_sub.csv'), label_sub, delimiter=',')

if __name__ == '__main__':
    subset_yelp()