import numpy as np;
from Contingency_Table import contingency_table, generate_contingency_tables;
from Measures import map_measures_to_indices;
from Ranks import ranks;

vals = np.array([1,2,11,100,1000,10000]);

#generates contingency tables with all f_ij's taking all possibile values in 'vals' array
tables = generate_contingency_tables(vals);

# maps measures to indices and outputs a dictionary and an array
(measures_dict, measures_arr) = map_measures_to_indices();

n_measures = len(measures_dict);

#initialize the scores matrix
scores_matrix = np.zeros(shape=(1296,n_measures));

# computes scores for each table and updates the scores matrix
for idx,table in enumerate(tables):
    t = contingency_table(table);
    t.compute_scores();
    scores_matrix[idx] = t.scores;
    
#computes the ranks class with the given scores
ranks_matrix = ranks(scores_matrix, measures_arr);

# ranks_matrix.remove_outliers([measures_dict['implication_index'],
#                               measures_dict['coverage'],
#                               measures_dict['prevalence']]);


## Clustering
# n_clusters = 3;

#prints indices of the measures in each cluster
# clusters = ranks_matrix.form_clusters(n_clusters);

#plots the heatmap for the spearman correlation matrix and dendrogram derived from spearman correlations
# ranks_matrix.visualize();