import pickle
import sys
sys.path.append('.')

from sbm import gen_random_sbm, est_sbm_params, from_sbm_params
from load_data import load_amazon_data, load_yelp_data, load_FDcomp_data, load_peer_review_data, load_elliptic_data

def get_all_sbm_params():
    dataset_names = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']
    datasets = [load_amazon_data(), load_yelp_data(), 
                    load_FDcomp_data(), load_peer_review_data(), load_elliptic_data()]

    res = {}

    for name, data in zip(dataset_names, datasets):
        A, labels, _ = data

        params = est_sbm_params(A, labels)
        print(f'{name} n0: {params[0]}, n1: {params[1]}, p00: {round(params[2], 4)}, p11: {round(params[3], 4)}, p01: {round(params[4], 4)}')

        params2 = from_sbm_params(params)
        print(f'{name} n0: {params2[0]}, d0: {params2[1]}, d1: {round(params2[2], 4)}, d01: {round(params2[3], 4)}, r: {round(params2[4], 4)}')


    return res

def generate_sbms(n_samples=10):
    A, labels, _ = load_amazon_data()
    sbm_params = [round(p, 4) for p in est_sbm_params(A, labels)]
    sbm_params[2] = sbm_params[2] /10.
    sbm_params[3] = sbm_params[3] /10.
    sbm_params[4] = sbm_params[4] /10.
    graphs = [gen_random_sbm(sbm_params) for i in range(n_samples)]
    amazon_sbms = {'graphs': graphs, 'sbm_params': tuple(sbm_params)}

    A, labels, _ = load_peer_review_data()
    sbm_params = [round(p, 4) for p in est_sbm_params(A, labels)]
    sbm_params[2] = sbm_params[2] /10.
    sbm_params[3] = sbm_params[3] /10.
    sbm_params[4] = sbm_params[4] /10.
    graphs = [gen_random_sbm(sbm_params) for i in range(n_samples)]
    peer_review_sbms = {'graphs': graphs, 'sbm_params': tuple(sbm_params)}

    return {'amazon_sbm': amazon_sbms, 'peer_review_sbm': peer_review_sbms}

if __name__ == '__main__':
    get_all_sbm_params()

    out = generate_sbms()
    pickle.dump(out, open('datasets/sbms.pkl', 'wb'))
    