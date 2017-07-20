import numpy as np

import analysis_functions as af
import IM_rank_correlations as IMR

def cluster_set_1():
    
    from Measures import form_measures_dict;
    rm = IMR.ranks(IMR.scores_matrix_dense, IMR.measures_arr);
    ma = rm.measures_arr;
    md = rm.measures_dict;
    rm.compute_correlation();
    
    def form_cluster_array(md):
        clusters = [];

        clusters.append([md['support'], md['prevalence'], md['least_contradiction'], 
                         md['example_counterexample_rate'], md['confirm_descriptive'], md['leverage'], 
                         md['confidence_causal'], md['confirmed_confidence_causal'], 
                         md['conviction'], md['zhang'], md['yules_y'], md['confirm_causal'], 
                         md['putative_causal_dependency'], md['klosgen'], md['dependency'], 
                         md['one_way_support'], md['goodman_kruskal'], md['accuracy'], 
                         md['cosine'], md['kulczynsky_1'], md['information_gain'], 
                         md['novelty'], md['two_way_support'], md['collective_strength'], 
                         md['kappa'], md['mutual_information'], md['j_measure'], 
                         md['gini_index'], md['normalized_mutual_information'], md['laplace_correction']]);
        
        clusters.append([md['recall'], md['negative_reliability'], md['relative_risk'],
                         md['coverage'], md['implication_index']]);
        clusters = np.array(clusters);
        return clusters
    
    clusters = form_cluster_array(md);
    
    clusters_flat = np.empty(0, int);
    
    for cluster in clusters:
        clusters_flat = np.append(clusters_flat, cluster);
    
    ma = ma[clusters_flat];
    md = form_measures_dict(ma);

    clusters = form_cluster_array(md);

    return (clusters, md, ma);