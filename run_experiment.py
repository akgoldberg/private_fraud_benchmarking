import load_data
import fraud_detectors
import sbm
from helpers import auc_score
import partition_aggregate
import numpy as np
import time
import pickle 
import matlab.engine

ENG = matlab.engine.start_matlab()
ENG.addpath('fraud_detector_implementations/telltail_matlab', nargout=0)

FRAUD_ALGOS = {
    'random': fraud_detectors.score_random,
    'clustering_coeff': lambda A: fraud_detectors.score_by_clustering_coeff(A),
    'clustering_coeff_neg': lambda A: fraud_detectors.score_by_clustering_coeff(A, negate=True),
    'community_detection': lambda A: fraud_detectors.score_by_community_detection(A),
    'truncate_svd_sum10': lambda A: fraud_detectors.score_by_truncate_svd(A, 10, agg_method='sum'),
    'truncate_svd_max10': lambda A: fraud_detectors.score_by_truncate_svd(A, 10, agg_method='max'),
    'degree': fraud_detectors.score_by_degree,
    'degree_neg': lambda A: fraud_detectors.score_by_degree(A, negate=True)
}

AGG_ALGOS = {
    'agg1': (['community_detection', 'truncate_svd_sum10', 'degree', 'clustering_coeff'], np.sum),
    'agg2': (['community_detection', 'degree_neg', 'truncate_svd_max10', 'clustering_coeff_neg'], np.sum),
    'agg_max': (['clustering_coeff', 'truncate_svd_sum10', 'truncate_svd_max10', 'degree', 'community_detection'], np.max)
}


# run all fraud detectors on data A and return AUC scores
def eval_all_fraud_detectors(A, labels, print_out=True):
    benign_ind = np.where(labels != 1)[0]
    fraud_ind = np.where(labels == 1)[0]

    scores = {}
    aucs = {}
    times = {}

    for name, algo in FRAUD_ALGOS.items():
        if print_out:
            print('Evaluating', name)
        t = time.time()
        S = algo(A)
        times[name] = time.time() - t
        scores[name] = S
        aucs[name] = auc_score(S, benign_ind, fraud_ind)
    
    for name, (algos, agg_func) in AGG_ALGOS.items():
        S = fraud_detectors.agg_scores([scores[algo] for algo in algos], agg_func=agg_func)
        scores[name] = S
        aucs[name] = auc_score(S, benign_ind, fraud_ind)

    return aucs, times
    
# evaluate all fraud detectors on all datasets non-privately
def run_evaluation_non_private():
    dataset_names = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']
    datasets = [load_data.load_amazon_data(), load_data.load_yelp_data(), 
                    load_data.load_FDcomp_data(), load_data.load_peer_review_data(), load_data.load_elliptic_data()]

    res = {}

    for name, data in zip(dataset_names, datasets):
        A, labels, metadata = data
        print(f'Running experiment on {name}')
        print(load_data.summarize_data(A, labels, None))

        aucs, runtimes = eval_all_fraud_detectors(A, labels, print_out=True)

        res[name] = {}
        res[name]['aucs'] = aucs
        res[name]['runtime'] = runtimes

    return res

def agg_dicts(dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = []
            res[k].append(v)
    return res

# k is num partitions, sub_rate is fraction of fraud vertices to subsample in each partition
def run_evaluation_subsample_aggregate(k, sub_rate):
    dataset_names = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']
    datasets = [load_data.load_amazon_data(), load_data.load_yelp_data(), 
                    load_data.load_FDcomp_data(), load_data.load_peer_review_data(), load_data.load_elliptic_data()]

    res = {}

    for name, data in zip(dataset_names, datasets):
        A, labels, _ = data
        print(f'Running experiment on {name}')
        print(load_data.summarize_data(A, labels, None))

        benign_ind = np.where(labels != 1)[0]
        fraud_ind = np.where(labels == 1)[0]

        subgraphs = partition_aggregate.partition_duplicate_graph(A, benign_ind, fraud_ind, k, sub_rate)

        all_aucs = []
        all_runtimes = []
        for i in range(k):
            A_sub, label_sub = subgraphs[i]
            aucs, runtimes = eval_all_fraud_detectors(A_sub, label_sub, print_out=False)
            all_aucs.append(aucs)
            all_runtimes.append(runtimes)

            if i % 10 == 0:
                print(f'Finished partition {i} of {k}')

        res[name] = {}
        res[name]['aucs'] = agg_dicts(all_aucs)
        res[name]['runtime'] = agg_dicts(all_runtimes)

    return res


if __name__ == '__main__':
    res = run_evaluation_non_private()
    # # res = run_evaluation_subsample_aggregate(100, 0.01)
    # # with open('results/pda_100_1.pkl', 'wb') as f:
    # #     pickle.dump(res, f)
    # res = run_evaluation_subsample_aggregate(100, 0.1)
    # with open('results/pda_100_10.pkl', 'wb') as f:
    #     pickle.dump(res, f)
    # # res = run_evaluation_subsample_aggregate(100, 0.5)
    # # with open('results/pda_100_50.pkl', 'wb') as f:
    # #     pickle.dump(res, f)
    # # res = run_evaluation_subsample_aggregate(100, 1.0)
    # # with open('results/pda_100_100.pkl', 'wb') as f:
    # #     pickle.dump(res, f)
    # # res = run_evaluation_non_private()
    # # with open('results/non_private.pkl', 'wb') as f:
    # #     pickle.dump(res, f)
