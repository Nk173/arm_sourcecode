import numpy as np

import analysis_functions as af
import IM_rank_correlations as IMR


def cluster_set_1():
    rm = IMR.ranks(IMR.scores_matrix_sparse, IMR.measures_arr);
    ma = rm.measures_arr;
    md = rm.measures_dict;
    rm.compute_correlation();
    
#     while find_redundant(ma, rm):
#         i = find_redundant(ma, rm);
#         md, ma = rm.remove_outliers(i);
    
    clusters = af.form_clusters(8, rm);
    
    misc_cluster1 = np.array([], int);
    misc_cluster1 = np.append(misc_cluster1, clusters[4]);
    misc_cluster1 = np.append(misc_cluster1, clusters[7]);
    
#     misc_cluster2 = np.array([], int);
#     misc_cluster2 = np.append(misc_cluster2, clusters[6]);
#     misc_cluster2 = np.append(misc_cluster2, clusters[7]);
    
    clusters_new = [];
    clusters_new.append(clusters[0]);
    clusters_new.append(clusters[1]);
    clusters_new.append(clusters[2]);
    clusters_new.append(clusters[3]);
    clusters_new.append(clusters[5]);
    clusters_new.append(clusters[6]);
    clusters_new.append(misc_cluster1);
#     clusters_new.append(misc_cluster2);

    clusters_new = np.array(clusters_new);

    return (clusters_new, md, ma);


def cluster_set_2():
    rm = IMR.ranks(IMR.scores_matrix_sparse, IMR.measures_arr);
    ma = rm.measures_arr;
    md = rm.measures_dict;
    rm.compute_correlation();
    
#     while find_redundant(ma, rm):
#         i = find_redundant(ma, rm);
#         md, ma = rm.remove_outliers(i);
    
    clusters = af.form_clusters(4, rm);
    
    misc_cluster1 = np.array([], int);
    misc_cluster1 = np.append(misc_cluster1, clusters[1]);
    misc_cluster1 = np.append(misc_cluster1, clusters[3]);
    
#     misc_cluster2 = np.array([], int);
#     misc_cluster2 = np.append(misc_cluster2, clusters[6]);
#     misc_cluster2 = np.append(misc_cluster2, clusters[7]);
    
    clusters_new = [];
    clusters_new.append(clusters[0]);
#     clusters_new.append(clusters[1]);
    clusters_new.append(clusters[2]);
#     clusters_new.append(clusters[3]);
#     clusters_new.append(clusters[4]);
#     clusters_new.append(clusters[5]);
#     clusters_new.append(clusters[6]);
    clusters_new.append(misc_cluster1);
#     clusters_new.append(misc_cluster2);

    clusters_new = np.array(clusters_new);
#     clusters_new = np.array(clusters);

    return (clusters_new, md, ma);


def cluster_set_3():
    rm = IMR.ranks(IMR.scores_matrix_sparse, IMR.measures_arr);
    ma = rm.measures_arr;
    md = rm.measures_dict;
    rm.compute_correlation();
    
    clusters = af.form_clusters(3, rm);
    
    return (clusters, md, ma);