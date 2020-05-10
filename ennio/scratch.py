import pandas as pd
import numpy as np


features_df = pd.read_csv('../data/features_resnet.zip', delimiter=',', header=None)
#features_df.to_csv('../data/features_resnet.zip', compression='zip',float_format='%.8f', index=None, header=None)

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']
N = train_triplets_df.shape[0]

for i in range(0, N, 500):
    print(len(np.unique(np.array(train_triplets_df)[:i,:])))

def filter_train(label):
    return np.apply_along_axis(lambda row: row if label in row else [np.NAN, np.NAN, np.NAN], 1, train_triplets_df)

print()