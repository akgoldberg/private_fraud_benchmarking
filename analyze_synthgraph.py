import numpy as np
import pandas as pd
import pickle
import sys
import ast

sys.path.append('synthetic_algos')
import synthetic_algos.truncate_graph as truncate_graph
from run_partition_agg_experiment import load_validation_data, load_test_data

SYNTHETIC_ALGOS = {
    'sbm_dp': 'SBM',
    'attr_graph_simp': 'AGM (No Triangles)',
    'attr_graph': 'AGM',
    'topmfilter': 'TopMFilter',
}

def get_param_error(param_name, params, params_true):
    if param_name == 'sbm_params':
        _,_,p0_est,_,p01_est = params[param_name]
        _,_,p0,_,p01 = params_true[param_name]
        return [p0_est - p0, p01_est - p01], [p0, p01]
    if param_name == 'degree_seq':
        hist_len = max(len(params[param_name]), len(params_true[param_name]))
        hist_est = np.bincount(params[param_name], minlength=hist_len)
        hist_true = np.bincount(params_true[param_name].astype(int), minlength=hist_len)
        return (hist_est - hist_true).tolist(), hist_true.tolist()
    else:
        return params[param_name] - params_true[param_name], params_true[param_name]


def load_synthgraph_param_estimation(eps, cutoff_rate):
    d = pickle.load(open(f'results/synthetic_stats_{int(eps)}{int(100*cutoff_rate)}.pkl', 'rb'))

    all_stat_names = set(np.hstack([list(d['data'][0]['yelp'][synthetic_algo]['params'].keys()) for synthetic_algo in SYNTHETIC_ALGOS.keys()]))

    rows = []
    for iter, data in d['data'].items():
        for dataset, stats in data.items():
            for synthetic_algo, s in stats.items():
                row = d['params'].copy()

                params = s['params']
                params_true = s['params_true']

                if params is None:
                    continue
                row['iter'] = iter
                row['dataset'] = dataset
                row['synthetic_algo'] = synthetic_algo

                for param in params.keys():
                    err, true = get_param_error(param, params, params_true)
                    row[param] = true 
                    row[param + '_err'] = err
                
                for stat in all_stat_names.difference(params.keys()):
                    row[stat] = np.nan
                
                if synthetic_algo == 'attr_graph_simp':
                    row['n_triangles'] = np.nan
                
                if synthetic_algo == 'topmfilter':
                    row['n_edges_flipped'] = params_true['n_flipped']
                else:
                    row['n_edges_flipped'] = np.nan

                rows.append(row)
    return pd.DataFrame(rows)

def get_n_truncated(cutoff_rate, split='val'):
    def get_n(A, labels, cutoff_rate):
        max_degree = max(A.sum(axis=0).A1[labels==0])
        deg_cutoff = int(max_degree * cutoff_rate)
        A_trunc, _ = truncate_graph.truncate_graph(A, labels, deg_cutoff)
        return A.shape[0] - A_trunc.shape[0]

    out = {}
    d = load_validation_data() if split == 'val' else load_test_data()

    for name, data in d.items():
        if name in ['amazon_sbm', 'peer_review_sbm']:
            n_trunc = int(np.mean([get_n(A, labels, cutoff_rate) for A, labels, _ in data]))
            out[name] = n_trunc
        else: 
            A, labels, _ = data
            n_trunc = get_n(A, labels, cutoff_rate)
            out[name] = n_trunc
    
    return out 

def main():
    for cutoff_rate in [1., 0.75, 0.5]:
        print(f'========CUTOFF RATE {cutoff_rate}===========')
        print(get_n_truncated(cutoff_rate, split='val'))

    res = []

    for eps in [ 1., 2., 5., 10.]:
        for cutoff_rate in [1., 0.75, 0.5, 0.25]:
            res.append(load_synthgraph_param_estimation(eps, cutoff_rate))

    out = pd.concat(res)
    out.to_csv('results/synthetic_param_estimation.csv', index=False, sep=',')


if __name__ == '__main__':
    main()    