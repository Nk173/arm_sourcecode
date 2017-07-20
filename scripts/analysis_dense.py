import numpy as np

import IM_rank_correlations as IMR
from Measures import map_measures_to_indices
import cluster_sets_dense as cluster_sets
import analysis_functions as af

#Preprocessing 
(measures_dict, measures_arr) = map_measures_to_indices()
(ranks_matrix, scores_matrix) = IMR.create_ranks_matrix(IMR.tables_dense, measures_arr)
ranks_matrix.compute_correlation()

cluster_sets_array = []
measures_dicts_array = []
measures_arrs_array = []

(cluster_set, measures_dict_, measures_arr_) = cluster_sets.cluster_set_1()
cluster_sets_array.append(cluster_set)
measures_dicts_array.append(measures_dict_)
measures_arrs_array.append(measures_arr_)

(cluster_set, measures_dict_, measures_arr_) = cluster_sets.cluster_set_2()
cluster_sets_array.append(cluster_set)
measures_dicts_array.append(measures_dict_)
measures_arrs_array.append(measures_arr_)

(cluster_set, measures_dict_, measures_arr_) = cluster_sets.cluster_set_3()
cluster_sets_array.append(cluster_set)
measures_dicts_array.append(measures_dict_)
measures_arrs_array.append(measures_arr_)

(cluster_set, measures_dict_, measures_arr_) = cluster_sets.cluster_set_4()
cluster_sets_array.append(cluster_set)
measures_dicts_array.append(measures_dict_)
measures_arrs_array.append(measures_arr_)

af.analysis_to_csv(cluster_sets_array, measures_dicts_array)
