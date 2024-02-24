import numpy as np
import pandas as pd
import scipy
import pickle
from sbm import from_sbm_params

### YELP FRAUD ###
# https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud
def load_yelp_data(PATH = 'datasets/yelp_dataset', kind='homo', subsetted=True, train=False):
    if not subsetted:
        assert kind in ('homo', 'net_rur', 'net_rtr', 'net_rsr')
        mat = scipy.io.loadmat(f'{PATH}/YelpChi.mat')
        A = mat[kind]
        metadata = mat['features']
        labels = mat['label'].reshape(-1) # 0 is benign, 1 is fraud, -1 is unknown
        return A, labels, metadata
    else:
        suffix = '_train' if train else ''
        A = scipy.sparse.load_npz(f'{PATH}/A_sub{suffix}.npz')   
        labels = np.loadtxt(f'{PATH}/labels_sub{suffix}.csv', delimiter=',')
        return A, labels, None

#### AMAZON FRAUD ###
def load_amazon_data(PATH = 'datasets/amazon_dataset', kind='homo'):
    assert kind in ('homo', 'net_upu', 'net_usu', 'net_uvu')
    mat = scipy.io.loadmat(f'{PATH}/amazon.mat')
    A = mat[kind]
    metadata = mat['features']
    labels = mat['label'].reshape(-1) # 0 is benign, 1 is fraud, -1 is unknown

    return A, labels, metadata

### FINANCE FRAUD ###
# https://github.com/split-gnn/splitgnn
def load_FDcomp_data(PATH = 'datasets/FDcomp_dataset', kind='homo'):
    assert kind in ('homo', 'invest', 'provide', 'sale')
    A = scipy.sparse.load_npz(f'{PATH}/A_{kind}.npz')    
    labels = np.loadtxt(f'{PATH}/labels.csv')
    metadata= scipy.sparse.load_npz(f'{PATH}/metadata.npz')

    return A, labels, metadata

### PEER REVIEW FRAUD ### 
# https://arxiv.org/pdf/2207.02303.pdf 
def load_peer_review_data(PATH = 'datasets/peer_review_synthetic_dataset'):
    # undirected graph nodes are papers edges are citations
    A =  scipy.sparse.load_npz(f'{PATH}/A.npz')
    labels = np.loadtxt(f'{PATH}/labels.csv')# 0 is benign, 1 is fraud, -1 is unknown
    return A.astype(float), labels, None

### BITCOIN FRAUD ### 
# https://www.kaggle.com/datasets/ellipticco/elliptic-data-set, https://arxiv.org/pdf/1908.02591.pdf 
def load_elliptic_data(PATH = 'datasets/elliptic_bitcoin_dataset', train=False):
    suffix = '_train' if train else ''
    # undirected graph nodes are transactions edges are bitcoin used in the same transaction
    A = scipy.sparse.load_npz(f'{PATH}/A{suffix}.npz')
    metadata = scipy.sparse.load_npz(f'{PATH}/metadata{suffix}.npz')
    labels = pd.read_csv(f'{PATH}/labels{suffix}.csv')['class'].values # 0 is benign, 1 is fraud, -1 is unknown
    return A, labels, metadata

### SBM DATA ###
def load_sbms(print_params=False):
    data = pickle.load(open('datasets/sbms.pkl', 'rb'))
    sbm1 = data['amazon_sbm']
    sbm2 = data['peer_review_sbm']
    graphs1 = [(A,labels,None) for A,labels in sbm1['graphs']]
    graphs2 = [(A,labels,None) for A,labels in sbm2['graphs']]

    if print_params:
        print('Amazon SBM params:', sbm1['sbm_params'])
        params2 = from_sbm_params(sbm1['sbm_params'])
        print(f'Amazon SBM params: n0: {params2[0]}, d0: {params2[1]}, d1: {round(params2[2], 4)}, d01: {round(params2[3], 4)}, r: {round(params2[4], 4)}')

        print('Peer Review SBM params:', sbm2['sbm_params'])
        params2 = from_sbm_params(sbm2['sbm_params'])
        print(f'Peer Review SBM params: n0: {params2[0]}, d0: {params2[1]}, d1: {round(params2[2], 4)}, d01: {round(params2[3], 4)}, r: {round(params2[4], 4)}')

    return graphs1, graphs2

def summarize_data(A, labels, metadata):
    return {
        'n_vertices':A.shape[0],
        'n_edges': A.nnz / 2.,
        'edge_density': round(100*A.nnz / (A.shape[0] * (A.shape[0] - 1)), 2),
        'n_fraud': sum(labels == 1),
        'max_degree': max(A.sum(axis=0).A1),
        'avg_degree': A.sum() / A.shape[0],
        'n_unknown': sum(labels == -1),
        'n_metadata_cols': metadata.shape[1] if metadata is not None else 0
    }

if __name__ == '__main__':
    d = {}
    # Load validation data and print summary statistics
    A, labels, metadata = load_yelp_data(train=True)
    d['yelp'] = summarize_data(A, labels, metadata)
    A, labels, metadata = load_elliptic_data(train=True)
    d['elliptic'] = summarize_data(A, labels, metadata)
    print('=== Validation Datasets ===')
    print(pd.DataFrame(d))
    # print data frame as latex table
    print(pd.DataFrame(d).to_latex(float_format="%.2f"))
    load_sbms(print_params=True)
    
    d = {}
    # Testing loading of data
    A, labels, metadata = load_yelp_data(train=False)
    d['yelp'] = summarize_data(A, labels, metadata)
    A, labels, metadata = load_amazon_data()
    d['amazon'] = summarize_data(A, labels, metadata)
    A, labels, metadata = load_FDcomp_data()  
    d['FDcomp'] = summarize_data(A, labels, metadata)
    A, labels, metadata = load_peer_review_data()
    d['peer_review'] = summarize_data(A, labels, metadata)
    A, labels, metadata = load_elliptic_data(train=False)
    d['elliptic'] = summarize_data(A, labels, metadata)
    print('=== Test Datasets ===')
    print(pd.DataFrame(d))
    print(pd.DataFrame(d).to_latex(float_format="%.2f"))


