import numpy as np
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from partition_aggregate import agg_mean_laplace, agg_median_inverse_sensitivity
from helpers import similarity_kendall_tau

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

ALGORITHM_NAMES = {
    'random': 'Random',
    'clustering_coeff': 'Clustering Coeff',
    'clustering_coeff_neg': 'Neg Clustering Coeff',
    'community_detection': 'Community Detection',
    'truncate_svd_sum10': 'SVD Error Sum',
    'truncate_svd_max50': 'SVD Error Max',
    'degree': 'Degree',
    'degree_neg': 'Neg Degree',
    'agg1': 'Agg1',
    'agg2': 'Agg2',
    'agg_max': 'AggMax'
}

DATASET_NAMES = {
    'yelp': 'Yelp',
    'elliptic': 'Elliptic',
    'amazon_sbm': 'Amazon (SBM)',
    'peer_review_sbm': 'Peer Review (SBM)',
    'amazon': 'Amazon',
    'FDcomp': 'FDcomp',
    'peer_review': 'Peer Review'
}

VAL_DATASETS = ['yelp', 'elliptic']
TEST_DATASETS = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']

def load_nonprivate_results(stat='aucs', split='val'):
    with open(f'results/{split}_non_private.pkl', 'rb') as f:
        res_non_private = pickle.load(f)
    
    dfs = []

    for i, res in res_non_private.items():
        for dataset, d in res.items():
            df = pd.DataFrame(d[stat], index=[0])
            df['dataset'] = dataset
            df['iter'] = i
            dfs.append(df)

    out = pd.concat(dfs).reset_index(drop=True)
    return out.groupby('dataset').mean().reset_index()

def generate_leaderboard(df, datasets, algos, reformat_dataset=True, print_latex=False):
    if reformat_dataset:
        df = df.melt(id_vars=['dataset'], value_vars=algos, var_name='algo', value_name='auc')
    dfs = []
    for dataset in datasets:
        df_sub = df[df['dataset'] == dataset].reset_index(drop=True)
        df_sub = df_sub.sort_values('auc', ascending=False).reset_index(drop=True)
        df_sub['rank'] = df_sub.index + 1
        dfs.append(df_sub)

    if print_latex:
        dfs_print = [df.replace({'algo': ALGORITHM_NAMES}) for df in dfs]
        dfs_print = [df.rename(columns={'algo': f'algo {df.dataset.unique()[0]}', 'auc': f'auc {df.dataset.unique()[0]}'}) for df in dfs_print]
        if len(datasets) <= 3:
            df_latex = pd.concat([df.set_index('rank').drop(columns='dataset') for df in dfs_print], axis=1)
            print(df_latex.to_latex(index=True, float_format="%.2f"))
        else:
            df_latex1 = pd.concat([df.set_index('rank').drop(columns='dataset') for df in dfs_print[:3]], axis=1)
            df_latex2 = pd.concat([df.set_index('rank').drop(columns='dataset') for df in dfs_print[3:]], axis=1)
            print(df_latex1.to_latex(index=True, float_format="%.2f"))
            print(df_latex2.to_latex(index=True, float_format="%.2f"))
    
    return pd.concat(dfs) 

def generate_nonprivate_leaderboard(split='val', print_latex=False):
    df = load_nonprivate_results(split=split)
    algos = df.columns[2:-1]
    return generate_leaderboard(df, VAL_DATASETS+['amazon_sbm', 'peer_review_sbm'] if split=='val' else TEST_DATASETS, algos, print_latex=print_latex)

##############################################################
## Analysis of the results of the SVD best rank experiments ##
##############################################################

def load_svd_results():
    with open('results/svd_threshold.pkl', 'rb') as f:
        res = pickle.load(f)
    dfs = []
    for dataset, d in res.items():
        df = pd.DataFrame(d) 
        df['dataset'] = dataset
        dfs.append(df)
    df = pd.concat(dfs)
    df.rename({'ranks': 'rank'}, axis=1, inplace=True)
    return df 

def plot_svd_results(df):
    # make 3 subplots for each dataset with 2 lines for each agg_method sum and max and rank on the x-axis
    g = sns.FacetGrid(df, col='dataset', height=4, aspect=1.5)
    g.map(sns.lineplot, 'rank', 'sum', label='sum', color='green', linestyle='dashed')
    g.map(sns.lineplot, 'rank', 'max', label='max')
    # show the argmax of the auc for each dataset as a red point on each respective plot
    for ax, dataset in zip(g.axes.flat, df['dataset'].unique()):
        df_sub = df[df['dataset'] == dataset]
        max_sum = df_sub[df_sub['sum'] == df_sub['sum'].max()]
        max_max = df_sub[df_sub['max'] == df_sub['max'].max()]
        ax.plot(max_sum['rank'], max_sum['sum'], 'ro')
        ax.plot(max_max['rank'], max_max['max'], 'ro')
        # print the max sum and max max on the plot
        ax.text(max_sum['rank'], max_sum['sum'], f'{max_sum["rank"].values[0]}, {max_sum["sum"].values[0]:.2f}', fontsize=12)
        ax.text(max_max['rank'], max_max['max'], f'{max_max["rank"].values[0]}, {max_max["max"].values[0]:.2f}', fontsize=12)
    g.add_legend(pos='lower center', fontsize=16)
    plt.savefig('results/figures/svd_threshold.png')

    d = df.groupby(['rank'])[['sum', 'max']].mean().reset_index()
    # get the rank with the highest mean auc for each agg_method
    print('Sum')
    print(d[d['sum'] == d['sum'].max()])
    print('Max')
    print(d[d['max'] == d['max'].max()])


################################################################
## Analysis of the subsample aggreagte grid search experiment ##
################################################################

def load_pda_validation_results(stat='aucs'):
    with open('results/val_pda.pkl', 'rb') as f:
        res_pda = pickle.load(f)
    dfs = []
    
    for (k, sub_rate), res_all in res_pda.items():
        for i, res in res_all.items():
            for dataset, d1 in res.items():
                df = pd.DataFrame(d1[stat])
                df['dataset'] = dataset
                df['sub_rate']  = sub_rate
                df['k'] = k
                df['iter'] = i
                dfs.append(df)
    # data frame with columns dataset, sub_rate, k, and column with list of aucs in each partition for each algo
    return pd.concat(dfs).groupby(['dataset', 'sub_rate', 'k', 'iter']).agg(list).reset_index()

def make_agg_df(df, algos, agg_method):
    df_agg = df.copy()
    for algo in algos:
        df_agg[algo] = np.array([agg_method(a) for a in df[algo]])
    df_agg['dataset'] = df['dataset']
    df_agg['sub_rate'] = df['sub_rate']
    df_agg['k'] = df['k']
    return df_agg

def analyze_pda_validation_results_leaderboard(stat='aucs', eps=1.0):
    df_nonpriv = load_nonprivate_results(split='val')
    df_pda = load_pda_validation_results()
    algos = df_nonpriv.columns[2:-1]
    print('Datasets:', df_pda.dataset.unique())

    # split eps between algos evenly
    laplace_aggs = make_agg_df(df_pda, algos, lambda a: agg_mean_laplace(a, eps/len(algos)))
    
    # for each combination of dataset, sub_rate, and k, for each iter get distance between non-private and private leaderboards
    nonpriv_leaderboard = generate_nonprivate_leaderboard(split='val')
    laplace_aggs = laplace_aggs.dropna()

    val_datasets = VAL_DATASETS+['amazon_sbm', 'peer_review_sbm']

    rows = []
    for sub_rate in laplace_aggs['sub_rate'].unique():
        for k in laplace_aggs['k'].unique():
            for iter in laplace_aggs['iter'].unique():
                priv_leaderboard = generate_leaderboard(laplace_aggs[(laplace_aggs['sub_rate'] == sub_rate) 
                                                                    & (laplace_aggs['k'] == k) & (laplace_aggs['iter'] == iter)], 
                                                                    val_datasets, algos)
                
                for dataset in val_datasets:
                    nonpriv_scores = nonpriv_leaderboard[nonpriv_leaderboard['dataset'] == dataset]
                    priv_scores = priv_leaderboard[priv_leaderboard['dataset'] == dataset]

                    if priv_scores.empty:
                        continue

                    algo_order = nonpriv_scores.algo
                    A = nonpriv_scores['rank'].values
                    B = priv_scores.set_index('algo').loc[algo_order]['rank'].values
                    scores = nonpriv_scores['auc'].values
                    dist = similarity_kendall_tau(A, B, scores)
                    rows += [{'k': k, 'sub_rate': sub_rate, 'iter': iter, 'dataset': dataset, 'dist': dist}]
    
    df = pd.DataFrame(rows)
    df['eps'] = eps
    df['method'] = 'laplace'
    return df

def get_leaderboard_baselines(split='val'):
    val = generate_nonprivate_leaderboard(split='val')
    test = generate_nonprivate_leaderboard(split='test')

    if split == 'val':
        print('=== Kendall Tau Baseline for Validation Sets ===')
        baselines = []
        for dataset in VAL_DATASETS + ['amazon', 'peer_review']:
            test_scores = test[test['dataset'] == dataset]
            if dataset in ['amazon', 'peer_review']:
                dataset = f'{dataset}_sbm'
            val_scores = val[val['dataset'] == dataset]

            algo_order = val_scores.algo
            A = val_scores['rank'].values
            B = test_scores.set_index('algo').loc[algo_order]['rank'].values
            scores = val_scores['auc'].values
            dist = similarity_kendall_tau(A, B, scores)
            print(f'Test vs. Val {dataset}: {round(dist, 4)}')

            dists = []
            for _ in range(1000):
                B = np.random.permutation(A)
                dists += [similarity_kendall_tau(A, B, scores)]
            
            print(f'Random: {round(np.mean(dists), 4)}')
            baselines.append(round(np.mean(dists), 4))
        
        return dict(zip(VAL_DATASETS + ['amazon_sbm', 'peer_review_sbm'], baselines))

    if split == 'test':
        baselines = []
        print('=== Kendall Tau Baseline for Test Sets ===')
        baselines = []
        for dataset in TEST_DATASETS:
            test_scores = test[test['dataset'] == dataset]

            A = test_scores['rank'].values
            scores = test_scores['auc'].values

            dists = []
            for _ in range(1000):
                B = np.random.permutation(A)
                dists += [similarity_kendall_tau(A, B, scores)]
            
            print(f'Random: {round(np.mean(dists), 4)}')
            baselines.append(round(np.mean(dists), 4))
    
   

def plot_pda_validation_results_leaderboard(eps):
    df = pd.read_csv('results/pda_validation_results_leaderboard.csv')
    df = df[~(df['sub_rate'] == 0.01)]
    df = df[df['eps'] == eps]
    # plot for all datasets and all sub_rates 
    markers = ['o', 's', 'v', 'D']
    palette = sns.color_palette()
    g = sns.relplot(data=df, x='sub_rate', y='dist', col='k', hue='dataset', kind='line', markers=markers, style='dataset',
                     height=4, aspect=1.5, col_wrap=3, dashes=False, lw=3, markersize=8, hue_order=VAL_DATASETS + ['amazon_sbm', 'peer_review_sbm'], palette=palette)
    g.add_legend(fontsize=16, loc='lower center')

    baselines = get_leaderboard_baselines()
    # add horizontal lines for baselines for each dataset
    for ax in g.axes.flat:
        for i, dataset in enumerate(VAL_DATASETS + ['amazon_sbm', 'peer_review_sbm']):
            # get color of line for that dataset in g
            color = palette[i]
            ax.axhline(baselines[dataset], ls='--', color=color)

    # set x-axis labels to be values of k from the data
    x_vals = df['sub_rate'].unique()
    g.set(xticks=x_vals)

    plt.savefig(f'results/figures/pda_validation_results_leaderboard_{eps}.png')



def analyze_pda_validation_results_oneshot(stat='aucs', eps=1.0, n_noise_draws=20):
    df_nonpriv = load_nonprivate_results(split='val')
    df_pda = load_pda_validation_results()
    algos = df_nonpriv.columns[2:-1]

    laplace_aggs = make_agg_df(df_pda, algos, lambda a: [agg_mean_laplace(a, eps) for _ in range(n_noise_draws)])
    # inv_sens_aggs = make_agg_df(df_pda, algos, lambda a: [agg_median_inverse_sensitivity(a, eps) for _ in range(n_noise_draws)])

    # get difference between non-private and private
    df_diff_laplace = df_nonpriv.merge(laplace_aggs, on=['dataset'], suffixes=('_nonpriv', '_priv'))
    # df_diff_inv_sens = df_nonpriv.merge(inv_sens_aggs, on=['dataset'], suffixes=('_nonpriv', '_priv'))

    # get bias and MSE for each algo
    for algo in algos:
        df_diff_laplace[f'bias_{algo}'] = df_diff_laplace[f'{algo}_priv'] - df_diff_laplace[f'{algo}_nonpriv']
        df_diff_laplace[f'MSE_{algo}'] = (df_diff_laplace[f'bias_{algo}']**2).apply(np.mean)
        df_diff_laplace[f'bias_{algo}'] = df_diff_laplace[f'bias_{algo}'].apply(np.mean)

        # df_diff_inv_sens[f'bias_{algo}'] = df_diff_inv_sens[f'{algo}_priv'] - df_diff_inv_sens[f'{algo}_nonpriv']
        # df_diff_inv_sens[f'MSE_{algo}'] = (df_diff_inv_sens[f'bias_{algo}']**2).apply(np.mean)
        # df_diff_inv_sens[f'bias_{algo}'] = df_diff_inv_sens[f'bias_{algo}'].apply(np.mean)

    stat_cols = [f'bias_{algo}' for algo in algos] + [f'MSE_{algo}' for algo in algos] 
    # aggregate over multiple iters   
    res_laplace = df_diff_laplace.groupby(['dataset', 'sub_rate', 'k'])[stat_cols].mean().reset_index()
    # res_inv_sens = df_diff_inv_sens.groupby(['dataset', 'sub_rate', 'k'])[stat_cols].mean().reset_index()

    res_laplace['method'] = 'laplace'
    res_laplace['eps'] = eps
    # res_inv_sens['method'] = 'inv_sens'
    # res_inv_sens['eps'] = eps
    return res_laplace
    # return pd.concat([res_laplace, res_inv_sens])


def plot_pda_validation_results(stat='bias', method='laplace', col='k', x='sub_rate', eps=1.0, dataset = 'yelp'):
    df = analyze_pda_validation_results_oneshot(eps=eps)
    df = df[df['method'] == method]
    stats = [c for c in df.columns if stat in c]
    if stat == 'bias':
        df = df[df.eps == df.eps.max()]
    if stat == 'MSE':
        df = df[df.eps == eps]
    
    df = df.melt(id_vars=['dataset', 'sub_rate', 'k', 'method', 'eps'], value_vars=stats, var_name='algo', value_name=stat)
    df['algo'] = ['_'.join(a.split('_')[1:]) for a in df['algo']]
    df.dropna(inplace=True)
    df = df[~df.algo.isin(['agg1', 'agg2', 'agg_max', 'random'])]
    df = df[df.sub_rate != 0.01]
    if stat == 'MSE':
        # use RMSE
        df[stat] = df[stat].apply(np.sqrt)
        # rename column MSE to RMSE
        df.rename(columns={stat: 'RMSE'}, inplace=True)
        stat = 'RMSE'

    # plot bias for each combination of dataset, sub_rate with all algorithms on one plot
    df = df[df['dataset'] == dataset]
    markers = ['o', 's', 'v', 'D', '<', 'd', '>']
    g = sns.relplot(data=df, x=x, y=stat, col=col, hue='algo', kind='line', markers=markers, style='algo',
                     height=4, aspect=1.5, col_wrap=3, dashes=False, lw=3, markersize=8)
    g.add_legend(fontsize=16, title='Fraud Detector', title_fontsize='16')
    
    # set x-axis labels to be values of k from the data
    x_vals = df[x].unique()
    g.set(xticks=x_vals)
    
    if stat == 'bias':
        lim = round(df[stat].abs().max(), 2)
        # make y limits symmetric
        g.set(ylim=(-lim, lim))
        # add dotted line at y=0
        for ax in g.axes:
            ax.axhline(0, ls='--', color='black')
    

    plt.savefig(f'results/figures/pda_validation_results_{dataset}_{stat}_{method}.png')
    
    # sns.set(style='whitegrid')
    # # plot bias for each combination of dataset, sub_rate, and k for epsilon = 1 with all algorithms on one plot
    # g = sns.FacetGrid(df, col='dataset', row='sub_rate', hue='method', height=4, aspect=1.5)
    # g.map(sns.lineplot, 'k', 'bias_pagerank', label='pagerank')

if __name__ == '__main__':
    get_leaderboard_baselines(split='val')
    get_leaderboard_baselines(split='test')
    # df_leaderboard = generate_nonprivate_leaderboard(split='val')
    # print('SBM LEADERBOARDS')
    # print(df_leaderboard[df_leaderboard.dataset == 'peer_review_sbm'])
    # print(df_leaderboard[df_leaderboard.dataset == 'amazon_sbm'])

    # get_leaderboard_baselines()
    # plot_pda_validation_results_leaderboard(eps=10*0.5)
    # plot_pda_validation_results_leaderboard(eps=10*1.0)
    # plot_pda_validation_results_leaderboard(eps=10*2.0)

    # plot_pda_validation_results(stat='MSE', method='laplace', col='k', x='sub_rate', eps=1.0, dataset = 'amazon_sbm')
    # plot_pda_validation_results(stat='bias', method='laplace', col='k', x='sub_rate', eps=1.0, dataset = 'amazon_sbm')

    # plot_pda_validation_results(stat='MSE', method='laplace', col='k', x='sub_rate', eps=1.0, dataset = 'peer_review_sbm')
    # plot_pda_validation_results(stat='bias', method='laplace', col='k', x='sub_rate', eps=1.0, dataset = 'peer_review_sbm')


    # generate_nonprivate_leaderboard(split='test', print_latex=True)
    
    # plot_pda_validation_results(dataset='yelp', stat='bias')
    # plot_pda_validation_results(dataset='yelp', stat='MSE')

    # plot_pda_validation_results(dataset='elliptic', stat='bias')
    # plot_pda_validation_results(dataset='elliptic', stat='MSE')

    # plot_pda_validation_results(dataset='peer review', stat='bias')
    # plot_pda_validation_results(dataset='peer review', stat='MSE')

    # df = pd.concat([analyze_pda_validation_results_oneshot(eps=eps) for eps in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5]])
    # df.to_csv('results/pda_validation_results_oneshot.csv')
    # df_nonpriv = load_nonprivate_results(split='val')
    # print(df_nonpriv.head())

    df = pd.concat([analyze_pda_validation_results_leaderboard(eps=10.*eps) for eps in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 1000.]])
    df.to_csv('results/pda_validation_results_leaderboard.csv')

    # df_pda = load_pda_validation_results()
    # print(df_pda.dataset.unique())
    ## Kendall Tau Baselines --> test vs train AND uniform random AUCs
