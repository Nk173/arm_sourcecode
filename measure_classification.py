import numpy as np;

from compute_invariance import compute_property_vectors;
import IM_rank_correlations as IMR;
from Measures import map_measures_to_indices;


def initialize_X_Y(measures_arr, property_dict):
    n_measures = len(measures_arr);
    n_properties = len(property_dict[measures_arr[0]]);
    X = np.zeros(shape=(n_measures, n_properties), dtype=int);
    Y = np.zeros(n_measures, dtype=int);
    return X,Y;

def assign_clusters_to_Y(clusters_arr, Y):
    for class_value,cluster_array in enumerate(clusters_arr):
        Y[cluster_array] = class_value;
    return Y

def form_X(property_dict, measures_dict, X):
    for key,value in property_dict.items():
        index = measures_dict[key];
        X[index] = value;
    return X;

def export_to_pdf(tree_classifier,n_clusters, n_measures):
    import pydotplus
    from sklearn import tree;
    dot_data = tree.export_graphviz(tree_classifier, out_file=None);
    graph = pydotplus.graph_from_dot_data(data=dot_data);
    filename = 'decision_tree_clusters_' + str(n_clusters) + '_measures_' + str(n_measures) + '.pdf';
    graph.write_pdf(filename);
    
def classify_decision_tree(X,Y, criterion='gini'):
    from sklearn import tree;
    tree_classifier = tree.DecisionTreeClassifier(criterion=criterion);
    tree_classifier.fit(X,Y);
    predictions = tree_classifier.predict(X)
    return predictions, tree_classifier;

#same as the number of classes to classify properties into
# n_clusters = 2;
# (measures_dict, measures_arr) = map_measures_to_indices();

#rank_matrix.ranks is mxn where m=rank_of_individual_table  and n=measure_index
# ranks_matrix = IMR.ranks_matrix;

#forming clusters
# clusters = ranks_matrix.form_clusters(n_clusters);

#dictionary with property array as values and measures names as keys
# property_dict = compute_property_vectors();

# X, Y = initialize_X_Y(measures_arr, property_dict);
# X = form_X(property_dict, measures_dict, X);
# Y = assign_clusters_to_Y(clusters, Y);

#outputs predictions and the classifier object
# predictions, classifier = classify_decision_tree(X,Y);

#export to pdf?
# export_to_pdf(classifier, n_clusters, len(X));
