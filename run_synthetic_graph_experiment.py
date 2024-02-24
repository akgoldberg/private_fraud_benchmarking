import numpy as np
import time 
import pickle
import sys
import os
import scipy
sys.path.append('synthetic_algos')
import pandas as pd
from joblib import Parallel, delayed

from synthetic_algos import attr_graph, sbm_dp, topmfilter
from run_partition_agg_experiment import load_validation_data, eval_all_fraud_detectors

def get_filenames(data_name, algo, i, deg_cutoff_rate, eps):
    A_file = f'synthetic_graphs/A_{data_name}_train_{algo}_{i}_{int(100*deg_cutoff_rate)}_{int(eps)}.npz'
    labels_file = f'synthetic_graphs/labels_{data_name}_train_{algo}_{i}_{int(100*deg_cutoff_rate)}_{int(eps)}.csv'
    return A_file, labels_file

def generate_synthetic_datasets(eps, deg_cutoff_rate, iters=10, non_private=False, statistics_only=False, save_data=False, run_parallel=False):
    d = load_validation_data()
    out = {}
    out['params'] = {'eps': eps, 'deg_cutoff_rate': deg_cutoff_rate, 'iters': iters, 'non_private': non_private}
    out['data'] = {}

    if run_parallel:
        args_list = [(d, eps, deg_cutoff_rate, i, statistics_only, non_private, save_data) for i in range(iters)]
        result_list = Parallel(n_jobs=2)(delayed(run_single_iter_generate_synthetic_datasets)(*args) for args in args_list)
        for data in result_list:
             out['data'][i] = data 
    else: 
        for i in range(iters):
            data = run_single_iter_generate_synthetic_datasets(d, eps, deg_cutoff_rate, i, statistics_only, non_private, save_data)
            out['data'][i] = data
    
    return out

def run_single_iter_generate_synthetic_datasets(d, eps, deg_cutoff_rate, i, statistics_only=False, non_private=False, save_data=False):
    print(f'===============Running iteration {i}======================')
        
    iter_out = {}

    for name, data in d.items():
        print(f'Running iter {i} on {name}')
        sys.stdout.flush()

        if name in ['amazon_sbm', 'peer_review_sbm']:
            data = data[i % len(data)]
        A, labels, _ = data
        
        # take max degree over benign nodes
        max_degree = max(A.sum(axis=0).A1[labels==0])
        deg_cutoff = int(max_degree * deg_cutoff_rate)

        # print(f'Running sbm_dp on {name}')
        A_name, labels_name = get_filenames(name, 'sbm_dp', i, deg_cutoff_rate, eps)
        if not statistics_only and os.path.exists(A_name) and os.path.exists(labels_name):
            # print('Data already exists, loading from file')
            A = scipy.sparse.load_npz(A_name)
            labels = np.loadtxt(labels_name, delimiter=',')
            graphs_sbm, params_sbm, params_sbm_true = [(A, labels)], None, None
            time_sbm = 0
        else:
            t = time.time()
            graphs_sbm, params_sbm, params_sbm_true = sbm_dp.run_generate_synthetic_sbm(A, labels, eps, deg_cutoff, non_private=non_private, n_samples=1, stats_only=statistics_only)
            time_sbm = time.time() - t

        if name == 'peer_review_sbm':
            # print('Not running AGM on peer_review_sbm, cannot sample effectively.')
            graphs_agm_simp, params_agm_simp, params_agm_simp_true = None, None, None
            graphs_agm, params_agm, params_agm_true = None, None, None
        else: 
            # print(f'Running attr_graph no triangles on {name}')

            A_name, labels_name = get_filenames(name, 'attr_graph_simp', i, deg_cutoff_rate, eps)
            if not statistics_only and os.path.exists(A_name) and os.path.exists(labels_name):
                # print('Data already exists, loading from file')
                A = scipy.sparse.load_npz(A_name)
                labels = np.loadtxt(labels_name, delimiter=',')
                graphs_agm_simp, params_agm_simp, params_agm_simp_true = [(A, labels)], None, None
                time_agm_simp = 0
            else:
                t = time.time()
                graphs_agm_simp, params_agm_simp, params_agm_simp_true = attr_graph.run_generate_synthetic_agm(A, labels, eps, deg_cutoff, non_private=non_private, n_samples=1, use_triangles=False, stats_only=statistics_only)
                time_agm_simp = time.time() - t

            # print(f'Running attr_graph on {name}')

            A_name, labels_name = get_filenames(name, 'attr_graph', i, deg_cutoff_rate, eps)
            if not statistics_only and os.path.exists(A_name) and os.path.exists(labels_name):
                # print('Data already exists, loading from file')
                A = scipy.sparse.load_npz(A_name)
                labels = np.loadtxt(labels_name, delimiter=',')
                graphs_agm, params_agm, params_agm_true = [(A, labels)], None, None
                time_agm = 0
            else:
                t = time.time()
                graphs_agm, params_agm, params_agm_true = attr_graph.run_generate_synthetic_agm(A, labels, eps, deg_cutoff, non_private=non_private, n_samples=1, use_triangles=True, stats_only=statistics_only)
                time_agm = time.time() - t
        
        # print(f'Running topmfilter on {name}')
        A_name, labels_name = get_filenames(name, 'topmfilter', i, deg_cutoff_rate, eps)
        if not statistics_only and os.path.exists(A_name) and os.path.exists(labels_name):
            # print('Data already exists, loading from file')
            A = scipy.sparse.load_npz(A_name)
            labels = np.loadtxt(labels_name, delimiter=',')
            graphs_topm, params_topm, params_topm_true = [(A, labels)], None, None
            time_topm = 0
        else:
            t = time.time()
            graphs_topm, params_topm, params_topm_true = topmfilter.run_generate_synthetic_topmfilter(A, labels, eps, deg_cutoff, non_private=non_private)
            time_topm = time.time() - t

        iter_out[name] = {'sbm_dp': {'params': params_sbm, 'params_true': params_sbm_true, 'time': time_sbm},
                    'attr_graph_simp': {'params': params_agm_simp, 'params_true': params_agm_simp_true, 'time': time_agm_simp},
                    'attr_graph': {'params': params_agm, 'params_true': params_agm_true, 'time': time_agm},
                    'topmfilter': {'params': params_topm, 'params_true': params_topm_true, 'time': time_topm}}

        print(f'Iter {i}: Generating synthetic graphs on {name} took {(time_agm + time_agm_simp + time_sbm + time_topm) / 60} minutes.')
        sys.stdout.flush()

        if not statistics_only: # evaluate AUCs
            all_aucs = []
            for algo, graphs in zip(['sbm_dp', 'attr_graph_simp', 'attr_graph', 'topmfilter'], [graphs_sbm, graphs_agm_simp, graphs_agm, graphs_topm]):
                if graphs is None:
                    continue
                # save data (assume only one sample each)
                A,labels = graphs[0]

                if save_data:
                    A_file, labels_file = get_filenames(name, algo, i, deg_cutoff_rate, eps)
                    if not os.path.exists(A_file) or not os.path.exists(labels_file):
                        # print('Saving data')
                        scipy.sparse.save_npz(A_file, A)
                        np.savetxt(labels_file, labels, delimiter=',')

                aucs, _ = eval_all_fraud_detectors(A, labels, print_out=False)
                aucs['eps'] = eps
                aucs['deg_cutoff_rate'] = deg_cutoff_rate
                aucs['dataset'] = name
                aucs['synthetic_algo'] = algo
                aucs['iter'] = i
                aucs['non_private'] = non_private
                aucs_df = pd.DataFrame(aucs, index=[0])

                # dump to csv each row
                aucs_file = f'synthetic_graphs/aucs_{int(100*deg_cutoff_rate)}_{int(eps)}_{i}.csv'
                if os.path.exists(aucs_file):
                    aucs_df.to_csv(aucs_file, mode='a', header=False, index=False)
                else:    
                    aucs_df.to_csv(aucs_file, index=False)

    return iter_out

def test_statistic_error(eps, deg_cutoff_rate, iters=10):
    d = load_validation_data()
    out = {}
    out['params'] = {'eps': eps, 'deg_cutoff_rate': deg_cutoff_rate, 'iters': iters}
    out['data'] = {}

    for i in range(iters):
        print(f'Running iteration {i}')
        
        iter_out = {}

        for name, data in d.items():
            if name in ['amazon_sbm', 'peer_review_sbm']:
                data = data[i % len(data)]
            A, labels, _ = data


def main():
    # get sufficient statistics for each algorithm
    # for eps in [1., 2., 5., 10.]:
    #     for cutoff_rate in [0.25]:
    #         print('=====================================================================')
    #         print('Running synthetic data stats generation for eps:', eps, 'cutoff_rate:', cutoff_rate)
    #         print('=====================================================================')
    #         out = generate_synthetic_datasets(eps, cutoff_rate, iters=10, non_private=False, statistics_only=True)
    #         pickle.dump(out, open(f'results/synthetic_stats_{int(eps)}{int(100*cutoff_rate)}.pkl', 'wb'))
    # return 


    # # run without privacy
    # out = generate_synthetic_datasets(0, 0, iters=10, non_private=True)
    # pickle.dump(out, open('results/synthetic_non_private.pkl', 'wb'))


    # run with privacy
    for eps in [5.]:
        cutoff_rate = 1.
        ### NEED TO USE BEST CUTOFF RATE FOR EACH EPS, DATASET
        print('=====================================================================')
        print('Running synthetic data generation for eps:', eps, 'cutoff_rate:', cutoff_rate)
        print('=====================================================================')
        out = generate_synthetic_datasets(eps, cutoff_rate, iters=10, non_private=False, run_parallel=False)
        pickle.dump(out, open(f'results/synthetic_{int(eps)}{int(100*cutoff_rate)}.pkl', 'wb'))

    # eps = 1.
    # cutoff_rate = 1.
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('results/synthetic11.pkl', 'wb'))

    # eps = 2.
    # cutoff_rate = 1.
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('results/synthetic21.pkl', 'wb'))

    # eps = 10.
    # cutoff_rate = 1.
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('results/synthetic101.pkl', 'wb'))

    #  eps = 1.
    # cutoff_rate = 0.75
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('datasets/synthetic1075.pkl', 'wb'))

    # eps = 1.
    # cutoff_rate = 0.5
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('datasets/synthetic1050.pkl', 'wb'))

    # eps = 2.
    # cutoff_rate = 1.
    # out = generate_synthetic_datasets(1., 1., iters=10, non_private=False)
    # pickle.dump(out, open('datasets/synthetic_non_private.pkl', 'wb'))


# generate synthetic data
if __name__ == '__main__':
    main()