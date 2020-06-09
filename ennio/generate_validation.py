import pandas as pd
import numpy as np

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']
N = train_triplets_df.shape[0]

for i in range(0, N, 500):
    print(len(np.unique(np.array(train_triplets_df)[:i,:])))



unique, counts = np.unique(np.array(train_triplets_df), return_counts=True)
dict_counts = dict(zip(unique,counts))


def get_rarest(N, dict):
    sorted_dict = sorted(dict.items(), key=lambda x: x[1])
    rarests = [a for a,b in sorted_dict[0:N]]
    return rarests

def filter_train(labels):
    #return np.apply_along_axis(lambda row: True if label in row else False, 1, np.array(train_triplets_df))
    return np.array([ id for id, row in train_triplets_df.iterrows()
                         if len([l for l in labels if l in row.values])>0])

rarests = get_rarest(100, dict_counts)
mask = filter_train(rarests)
filtered = train_triplets_df.iloc[mask]
filtered.to_csv("../data/val_triplets.txt", sep=' ', header=None)
print('we')