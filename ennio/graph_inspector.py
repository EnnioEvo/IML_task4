import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length

##read previous graph
#lengths = pd.read_csv("../data/lengths.csv", index_col=0)


# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']

triplets = np.array(train_triplets_df)
N = train_triplets_df.shape[0]
relations = np.zeros([N, 3])
relations[:N, 2] = 1  # similiar
# relations[N:,2] = 10 #different
relations[:N, 0:2] = triplets[:, [0, 1]]
# relations[N:,0:2] = triplets[:,[0,2]]

relations = pd.DataFrame(data=relations, columns=['S', 'D', 'W'])
relations.to_csv('../data/graph.csv')

N_triplets = 1000

G = nx.Graph()
G.add_nodes_from(np.unique(np.array(relations)[:N_triplets]))

for index, row in relations.iterrows():
    G.add_edge(row['S'], row['D'], weight=row['W'])
    if index == N_triplets - 1:
        break

lengths_gen = all_pairs_dijkstra_path_length(G, cutoff=5)
lengths = dict(lengths_gen)
lengths_df = pd.DataFrame.from_dict(data=lengths, orient='index')
sorted_columns = list(map(str,sorted(list(map(float,lengths_df.columns)))))
lengths_df = lengths_df.reindex(sorted_columns, axis=1)
lengths_df.to_csv('../data/lengths2.csv', header=True)
print()
