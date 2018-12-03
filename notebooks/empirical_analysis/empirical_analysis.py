import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.utils import gen_even_slices
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import numpy as np


path = '/Users/abhisheksharma/Desktop/KAIS/Scripts/Clustering Measures/Rules/'
tables = np.loadtxt(path+'rules_022_spambase.data.csv', skiprows=1,delimiter=',', dtype=int)

# def cluster_tables(k, tables):
#     metric = 'euclidean'
#     n_clusters = k
#     ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity=metric)
#     clusters = ac.fit_predict(tables[:,[0,3]])
#     cost = 0
#     print(k, clusters)
#     for c in range(n_clusters):
#         X = tables[:,[0,3]][clusters == c]
#         Y = tables[:,[0,3]][clusters == c]
#         n_jobs = 4
#         dist = Parallel(n_jobs=n_jobs, backend='threading')(delayed(euclidean_distances)(X, Y[s]) for s in gen_even_slices(Y.shape[0], n_jobs))
#         dist_matrix = np.hstack(dist)
#         # dist_matrix = euclidean_distances(tables[:,[0,3]][clusters == c])
#         cost += np.triu(dist_matrix, k=0).sum()
#     # Search for dense clusters
#     print(k, cost)

#     print('Dense clusters', [(cname, len(tables[clusters==cname]), (tables[clusters==cname,3]/tables[clusters==cname,0]).mean())
#                                 for cname in range(n_clusters)
#                                 if (tables[clusters==cname,3]/tables[clusters==cname,0]).mean() < 1 ])
#     print('Sparse clusters', [(cname, len(tables[clusters==cname]), (tables[clusters==cname,3]/tables[clusters==cname,0]).mean())
#                                 for cname in range(n_clusters)
#                                 if (tables[clusters==cname,3]/tables[clusters==cname,0]).mean() > 2 ])
#     print('')
#     plt.scatter(k, cost)

# # os.environ['MKL_NUM_THREADS'] = str(2)
# # Parallel(n_jobs=4, backend='threading')(delayed(cluster_tables)(k, tables) for k in range(2,20))
# for k in range(5,10):
#     cluster_tables(k, tables)


def cluster_tables(k, tables):
    metric = 'euclidean'
    n_clusters = k
    km = KMeans(n_clusters=n_clusters, n_jobs=-1)
    clusters = km.fit_predict(tables[:,[0,3]])
    cost = km.inertia_

    print(k, cost)

    # Search for dense clusters
    print('Dense clusters', [(cname, len(tables[clusters==cname]), (tables[clusters==cname,3]/tables[clusters==cname,0]).mean())
                                for cname in range(n_clusters)
                                if (tables[clusters==cname,3]/tables[clusters==cname,0]).mean() < 1 ])
    # Search for sparse clusters
    print('Sparse clusters', [(cname, len(tables[clusters==cname]), (tables[clusters==cname,3]/tables[clusters==cname,0]).mean())
                                for cname in range(n_clusters)
                                if (tables[clusters==cname,3]/tables[clusters==cname,0]).mean() > 2 ])
    print('')
    plt.scatter(k, cost)

# os.environ['MKL_NUM_THREADS'] = str(2)
# Parallel(n_jobs=4, backend='threading')(delayed(cluster_tables)(k, tables) for k in range(2,20))
for k in range(1,20):
    cluster_tables(k, tables)

plt.xticks(range(4,20))
plt.show()