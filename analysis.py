import numpy as np
import pickle 
import pandas as pd

def load_results():
    with open('results/non_private.pkl', 'rb') as f:
        res_non_private = pickle.load(f)
    with open('results/pda_100_1.pkl', 'rb') as f:
        res1 = pickle.load(f)
    with open('results/pda_100_10.pkl', 'rb') as f:
        res10 = pickle.load(f)
    with open('results/pda_100_50.pkl', 'rb') as f:
        res50 = pickle.load(f)
    with open('results/pda_100_100.pkl', 'rb') as f:
        res100 = pickle.load(f)

    return res_non_private, res1, res10, res50, res100

if __name__ == '__main__':
    dataset_names = ['amazon', 'yelp', 'FDcomp', 'peer_review', 'elliptic']
    res_non_private, res1, res10, res50, res100 = load_results()
    for d in dataset_names:
        print(d)
        print(pd.DataFrame(res_non_private[d]))