import load_data
import fraud_detectors
import sbm
from helpers import auc_score, permute_vetices
import partition_aggregate
import numpy as np
import time
import pickle 

FRAUD_ALGOS = {
    'random': fraud_detectors.score_random,
    'clustering_coeff': lambda A: fraud_detectors.score_by_clustering_coeff(A),
    'clustering_coeff_neg': lambda A: fraud_detectors.score_by_clustering_coeff(A, negate=True),
    'community_detection': lambda A: fraud_detectors.score_by_community_detection(A),
    'truncate_svd_sum10': lambda A: fraud_detectors.score_by_truncate_svd(A, 10, agg_method='sum'),
    'truncate_svd_max50': lambda A: fraud_detectors.score_by_truncate_svd(A, 50, agg_method='max'),
    'degree': fraud_detectors.score_by_degree,
    'degree_neg': lambda A: fraud_detectors.score_by_degree(A, negate=True)
}

AGG_ALGOS = {
    'agg1': (['community_detection', 'truncate_svd_sum10', 'degree', 'clustering_coeff'], np.sum),
    'agg2': (['community_detection', 'degree_neg', 'truncate_svd_max50', 'clustering_coeff_neg'], np.sum),
    'agg_max': (['clustering_coeff', 'truncate_svd_sum10', 'truncate_svd_max50', 'degree', 'community_detection'], np.max)
}

######################################################
####################### Helpers ######################
######################################################

# load validation data based on split of data
def load_validation_data():
    graphs1, graphs2 = load_data.load_sbms()
    return {'yelp': load_data.load_yelp_data(train=True),
             'elliptic': load_data.load_elliptic_data(train=True),
             'amazon_sbm': graphs1,
             'peer_review_sbm': graphs2}

def load_test_data():
    dataset_names = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']
    datasets = [load_data.load_amazon_data(), load_data.load_yelp_data(train=False), 
                    load_data.load_FDcomp_data(), load_data.load_peer_review_data(), load_data.load_elliptic_data(train=False)]
    
    return dict(zip(dataset_names, datasets))

# run all fraud detectors on data A and return AUC scores
def eval_all_fraud_detectors(A, labels, print_out=True):
    # randomly permute vertex labels
    A, labels = permute_vetices(A, labels)

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


def agg_dicts(dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = []
            res[k].append(v)
    return res

################################################################
################# Run Algorithms Non-Privately #################
################################################################

# evaluate all fraud detectors on all datasets non-privately
def run_evaluation_non_private(d, niters=10):
    out = {}

    for i in range(niters):
        res = {}
        print(f'Running iteration {i}')
        for name, data in d.items():
            if name in ['amazon_sbm', 'peer_review_sbm']:
                data = data[i % len(data)]
            A, labels, _ = data
            print(f'Running experiment on {name}')
            print(load_data.summarize_data(A, labels, None))

            aucs, runtimes = eval_all_fraud_detectors(A, labels, print_out=True)

            res[name] = {}
            res[name]['aucs'] = aucs
            res[name]['runtime'] = runtimes
        
        out[i] = res

    return out

################################################################
################# Choose SVD Rank Analysis #####################
################################################################
def grid_search_svd_threshold():
    d = load_validation_data()

    res = {}

    for name, (A, labels, _) in d.items():
        res[name] = {}
        print(f'Running svd on {name}')

        benign_ind = np.where(labels != 1)[0]
        fraud_ind = np.where(labels == 1)[0]

        ranks = [1, 5, 10, 20, 50, 100, 200, 500]
        sum_scores = []
        max_scores = []

        for rank in ranks:
            scores = fraud_detectors.score_by_truncate_svd(A, rank, agg_method='sum')
            print(f'Agg sum, Rank {rank}, AUC: {auc_score(scores, benign_ind, fraud_ind)}')
            sum_scores += [auc_score(scores, benign_ind, fraud_ind)]


            scores = fraud_detectors.score_by_truncate_svd(A, rank, agg_method='max')
            print(f'Agg max, Rank {rank} AUC: {auc_score(scores, benign_ind, fraud_ind)}')
            max_scores += [auc_score(scores, benign_ind, fraud_ind)]
        
        res[name]['sum'] = sum_scores
        res[name]['max'] = max_scores
        res[name]['ranks'] = ranks 

    with open('results/svd_threshold.pkl', 'wb') as f:
        pickle.dump(res, f)

#############################################################################
################# Choose Subsample Aggregate Parameters #####################
#############################################################################

def grid_search_partition_aggregate_params():
    d = load_validation_data()
    
    out = {}

    for k in [5, 10, 20, 50, 100, 200]:
        for sub_rate in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
            if sub_rate < 1./k:
                continue
            res = run_evaluation_subsample_aggregate(d, k, sub_rate)
            out[k, sub_rate] = res

    with open(f'results/val_pda.pkl', 'wb') as f:
        pickle.dump(out, f)

# k is num partitions, sub_rate is fraction of fraud vertices to subsample in each partition
def run_evaluation_subsample_aggregate(d, k, sub_rate, iters=10):
    print('=======Running with k:', k, 'sub_rate=========', sub_rate)
    
    out = {}

    for iter in range(iters):
        print(f'Running iteration {iter}')
        res = {}

        for name, data in d.items():
            if name in ['amazon_sbm', 'peer_review_sbm']:
                data = data[iter % len(data)]
            A, labels, _ = data
            A, labels = permute_vetices(A, labels)

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

            res[name] = {}
            res[name]['aucs'] = agg_dicts(all_aucs)
            res[name]['runtime'] = agg_dicts(all_runtimes)

            out[iter] = res

    return out


################################################################
################# Main Loop: Run Experiments #####################
################################################################
def main():
    
    # run non-private evaluations on val data
    d = load_validation_data()
    res = run_evaluation_non_private(d)
    with open('results/val_non_private.pkl', 'wb') as f:
        pickle.dump(res, f)
    
    return 
    # run grid search
    grid_search_partition_aggregate_params() 

    # run non-private evaluations on test data
    d = load_test_data()
    res = run_evaluation_non_private(d)
    with open('results/test_non_private.pkl', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    main()