import pandas as pd
import numpy as np
import networkx as nx
import scipy

def to_sparse(A):
    return scipy.sparse.csr_matrix(A).astype(float)

PATH = 'elliptic_bitcoin_dataset/original'

def make_clean_dataset(timestep=1, OUT_DIR='elliptic_bitcoin_dataset'):
    dfClasses = pd.read_csv(f'{PATH}/elliptic_txs_classes.csv')
    dfEdgelist= pd.read_csv(f'{PATH}/elliptic_txs_edgelist.csv')
    dfFeatures= pd.read_csv(f'{PATH}/elliptic_txs_features.csv')
    
    dfFeatures.columns = ['id', 'time'] + [f'local_{i}' for i in range(93)] + [f'aggregate_{i}' for i in range(72)]
    # subset to timestep 1
    df1 = dfFeatures[dfFeatures.time.isin([timestep])]
    # merge class labels with subset dataframe
    df = pd.merge(df1, dfClasses, left_on='id', right_on='txId', how='left')
    # map benign to 0, fraud to 1, unknown to -1
    df['class'] = df['class'].replace({'unknown': -1, '2': 0}).astype(int)

    # make adjacency matrix
    edges = dfEdgelist[dfEdgelist.txId1.isin(df.id) & dfEdgelist.txId2.isin(df.id)]
    G = nx.from_pandas_edgelist(edges, 'txId1', 'txId2')
    # Get the adjacency matrix
    A = nx.to_numpy_array(G, dtype=int)
    nodes = list(G.nodes())
    
    # make meta-data, remove ``aggregate`` features because they include info about other nodes
    metadata = df.set_index('id').loc[nodes].drop(columns=['class', 'txId', 'time']
                                   +[f'aggregate_{i}' for i in range(72)]).reset_index(drop=True)
    # make labels
    labels = df.set_index('id').loc[nodes]['class'].reset_index(drop=True)
    
    # save data to files
    scipy.sparse.save_npz(f'{OUT_DIR}/A.npz', to_sparse(A))
    scipy.sparse.save_npz(f'{OUT_DIR}/metadata.npz', to_sparse(metadata))
    labels.to_csv(f'{OUT_DIR}/labels.csv')


if __name__ == "__main__":
    make_clean_dataset()