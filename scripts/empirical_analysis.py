import IM_rank_correlations as IMR
from Measures import map_measures_to_indices
import analysis_functions as af

import numpy as np
import csv


# dataset = 'mushroom'
# dataset = 'adult'
# dataset = 'synthetic_dense'
dataset = 'synthetic_sparse'

# Setting appropriate parameters for each dataset
# Make sure the variable dataset is defined
if dataset == 'mushroom':
    data_path = '../rules_mushroom.csv'
    n_clusters = 20
    cluster_sets = 4
    idx = 0
    # Load the rules
    tabs = af.import_data(data_path)
    c_tables = af.agglomerative_clustering_dataset(tabs, n_clusters, idx)
elif dataset == 'adult':
    data_path = '../rules_adult_new.csv'
    cluster_sets = 2
    n_clusters = 100
    idx = 22
    tabs = af.import_data(data_path)
    c_tables = af.agglomerative_clustering_dataset(tabs, n_clusters, idx)
elif dataset == 'synthetic_sparse':
    cluster_sets = 3
    c_tables = IMR.tables_sparse
elif dataset == 'synthetic_dense':
    cluster_sets = 3
    c_tables = IMR.tables_dense

if len(c_tables):
    # Initialiazing and calculating scores and ranking for each measure
    (measures_dict, measures_arr) = map_measures_to_indices()
    (ranks_matrix, scores_matrix) = IMR.create_ranks_matrix(c_tables, measures_arr)
    ranks_matrix.compute_correlation()


    (clusters_2, measures_dict_2, measures_arr_2) = af.cluster_set_2(scores_matrix, measures_arr, cluster_sets)

    af.return_homogeneity(clusters_2, measures_dict_2)
