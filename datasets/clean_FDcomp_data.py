import dgl
import numpy as np
import scipy

def to_sparse(A):
    return scipy.sparse.csr_matrix(A).astype(float)

def make_clean_dataset(PATH = 'FDcomp_dataset'):
    g = dgl.load_graphs(f'{PATH}/original/comp.dgl')[0][0]
    A_homo = to_sparse(g.adjacency_matrix('homo').to_dense().numpy())
    A_invest = to_sparse(g.adjacency_matrix('invest_bc2bc').to_dense().numpy())
    A_provide = to_sparse(g.adjacency_matrix('provide_bc2bc').to_dense())
    A_sale = to_sparse(g.adjacency_matrix('sale_bc2bc').to_dense().numpy())
    labels = g.ndata['label'].numpy()
    metadata = to_sparse(g.ndata['feature'].numpy())
    train_mask, test_mask, valid_mask = g.ndata['train_mask'].numpy(), g.ndata['test_mask'].numpy(), g.ndata['valid_mask'].numpy()

    # save data to files
    scipy.sparse.save_npz(f'{PATH}/A_homo.npz', A_homo)
    scipy.sparse.save_npz(f'{PATH}/A_invest.npz', A_invest)
    scipy.sparse.save_npz(f'{PATH}/A_provide.npz', A_provide)
    scipy.sparse.save_npz(f'{PATH}/A_sale.npz', A_sale)
    scipy.sparse.save_npz(f'{PATH}/metadata.npz', metadata)
    np.savetxt(f'{PATH}/labels.csv', labels, delimiter=',')
    np.savetxt(f'{PATH}/train_mask.csv', train_mask, delimiter=',')
    np.savetxt(f'{PATH}/test_mask.csv', test_mask, delimiter=',')
    np.savetxt(f'{PATH}/valid_mask.csv', valid_mask, delimiter=',')

if __name__ == "__main__":
    make_clean_dataset()