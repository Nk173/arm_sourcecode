import IM_rank_correlations as IMR
import compute_invariance as compute_invariance

import numpy as np
import csv
from sklearn.cluster import AgglomerativeClustering

def import_data(data_path):
    tables = [];
    with open(data_path) as csvfile:
        rule_reader = csv.reader(csvfile)
        for rule in rule_reader:
            tables.append([int(rule[0]), int(rule[2]), int(rule[3]), int(rule[1])])
            
    return(np.array(tables))

#prints the measures list in each of the clusters
def print_cluster_set(clusters, ma):
    for idx, cluster in enumerate(clusters):
        print(len(cluster), cluster);        
        print(ma[cluster]);

def agglomerative_clustering_dataset(c_tables, n_clusters, idx):
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = ac.fit_predict(c_tables[:,1:4])
    tabs = c_tables[clusters==idx]
    return(tabs)

def form_clusters(n_clusters, ranks_matrix):
    clusters = ranks_matrix.form_clusters(n_clusters);
    return clusters

#forms a cluster vector corresponding to the lengths of the clusters
def form_cluster_set(clusters):
    cluster_vector = np.empty(len(clusters), int);
    for idx, cluster in enumerate(clusters):
        cluster_vector[idx] = len(cluster);
    return cluster_vector;

def cluster_set_2(scores_matrix, measures_arr, cluster_sets):
    rm = IMR.ranks(scores_matrix, measures_arr);
    ma = rm.measures_arr;
    md = rm.measures_dict;
    rm.compute_correlation();
    
    clusters = form_clusters(cluster_sets, rm);
    return (clusters, md, ma);

def analysis_to_csv(cluster_sets, measures_dicts):

    print_flag = True
    for cluster_set_id, cluster_set in enumerate(cluster_sets):
        
        (properties_array, property_names, support_array, entropy_array) = compute_invariance.compute_property_vectors(measures_dicts[cluster_set_id])
        cluster_property_array = compute_invariance.compute_property_frequencies_in_cluster_set_updated(properties_array, cluster_set);
        cluster_vector = form_cluster_set(cluster_set);
        
        print('# measures per cluster', cluster_vector, end='\n\n');

        # check if print_array already exists
        if not 'print_array' in locals():
            print_array = [[0]*0 for i in range(len(property_names))]

        (n_properties, n_prop_states, n_clusters) = cluster_property_array.shape

        for prop_id, property_vector in enumerate(cluster_property_array):
            # property_vector is (n_prop_states, n_clusters)
            homo = compute_invariance.compute_homogeneity(property_vector, cluster_vector)

            #need to print entropy, support etc only once
            if print_flag:
                print_array[prop_id].append(property_names[prop_id])
                print_array[prop_id].append([int(np.sum(property_vector[1-i,:])) for i in range(n_prop_states)])
                print_array[prop_id].append(support_array[prop_id,1])
                print_array[prop_id].append(entropy_array[prop_id])

            
            print_array[prop_id].append('')
            print_array[prop_id].append([ int(property_vector[1,j]) for j in range(n_clusters) ])
            print_array[prop_id].append(homo)
            print_array[prop_id].append(homo/entropy_array[prop_id])
        
        print_flag = False

    with open('analysis.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Property', 'Original Split ([Y, N])',	'Support', 'Entropy (normalized)'] + ['','Property Split','Homogeneity','Outlier']*len(cluster_sets))
        
        for prop_id, print_row in enumerate(print_array):
            writer.writerow(print_row)


                            
def return_homogeneity(cluster_set, measures_dict):    

    (properties_array, property_names, support_array, entropy_array) = compute_invariance.compute_property_vectors(measures_dict)

    cluster_property_array = compute_invariance.compute_property_frequencies_in_cluster_set_updated(properties_array, cluster_set);
    cluster_vector = form_cluster_set(cluster_set);
    print('# measures per cluster', cluster_vector, end='\n\n');

    (n_properties, n_prop_states, n_clusters) = cluster_property_array.shape

    for idx, property_vector in enumerate(cluster_property_array):
        # property_vector is (n_prop_states, n_clusters)

        homo = compute_invariance.compute_homogeneity(property_vector, cluster_vector)
        
        # print(property_vector)
        
        # Print Property name
        print(property_names[idx]);
    
        # Original split of properties into [Y,N]
        print([int(np.sum(property_vector[1-i,:])) for i in range(n_prop_states)])

        # Partitioning of Y's in the cluster set
        print([ int(property_vector[1,j]) for j in range(n_clusters) ])

        # Partitioning of N's in the cluster set
        print([ int(property_vector[0,j]) for j in range(n_clusters) ])

        print(homo)

        print('')