import numpy as np;
from Invariance import invariance;
from Measures import map_measures_to_indices;
import itertools;
import scipy.misc as sm;

(measures_dict, measures_arr) = map_measures_to_indices();

inv = ['P1', 'P2', 'P3', 'O1', 'O2', 'O3', 'O4', 'O5', 'US'];

for i,j in itertools.combinations(range(9),2):
    prop_name = inv[i] + ' and ' + inv[j];
    inv.append(prop_name);
    
    prop_name = inv[i] + ' or ' + inv[j];
    inv.append(prop_name);



# takes a property array and returns support array and corresponding measure of impurity
def compute_supports(properties_vector):
    (n_measures, n_props) = properties_vector.shape;    
    
    support_array = np.empty(n_props, float);
    entropy_array = np.empty(n_props, float);

    for i in range(n_props):
        n = np.sum(properties_vector[:,i]);
        N = len(properties_vector[:,i]);
        support_array[i] =  n/N ;
        entropy_array[i] = entropy(np.array([n, N-n]));

    return (support_array, entropy_array);

def convert_property_array_to_dict(properties_vector, measures_dict):

    properties_dict = dict();
    for key,value in measures_dict.items():
        properties_dict[key] = properties_vector[value];
    
    return properties_dict;


#Converts the 0/1 binary array to -1/+1 array
def convert_zero_to_minus_one(properties_vector):
    
    # properties_vector_modified = dict();

    for idx, value in enumerate(properties_vector):
        for idx_1, element in enumerate(value):
            if element:
                properties_vector[idx,idx_1] = 1;
            else:
                properties_vector[idx,idx_1] = -1;
    
    # for key, value in properties_vector.items():
    #     property_vector = np.empty(len(value), int);
    #     for idx, element in enumerate(value):
    #         if element:
    #             property_vector[idx] = 1;
    #         else:
    #             property_vector[idx] = -1;                

    #     properties_vector_modified[key] = property_vector;

    return properties_vector


# takes a boolean column of property vector (n_measures x 1) and threshold value and returns true if it is satisfied
def check_support_threshold(property_vector, threshold_lower, threshold_upper):
    support = np.sum(property_vector) / len(property_vector);
    
    if (support >= threshold_lower) and (support <= threshold_upper):
        return True;
    
    return False;

def compute_property_vectors_with_support(properties_vector, property_names, threshold_lower, threshold_upper = 1):
    
    # properties_vector_with_support = np.empty(0,bool);
    properties_vector_with_support = properties_vector;
    (n_measures, n_props) = properties_vector.shape;
    
    to_remove = np.empty(0, int);
    for i in range(n_props):
        if not (check_support_threshold(properties_vector[:,i], threshold_lower, threshold_upper)):
            to_remove = np.append(to_remove, i);
    
    properties_vector_with_support = np.delete(properties_vector_with_support, to_remove, axis=1);
    property_names = np.delete(property_names, to_remove, axis=0);

    return (properties_vector_with_support, property_names);


# Can take a dict of measure names as keys if needed, else will pull the original one in Measures module
# returns a dict with property vectors as values
def compute_property_vectors_without_support(measures_dict=measures_dict):
    n_props = 9 + (2 * sm.comb(9,2));
    # n_props = 7 + (2 * sm.comb(7,2)) + (4 * sm.comb(7,3)) + (9 * sm.comb(7,4)) + (16 * sm.comb(7,5));
    n_measures = len(measures_dict);

    properties_vector = np.empty((n_measures, int(n_props)));
    properties_vector_names = np.empty(int(n_props), object);

    for key,value in measures_dict.items():
        invariance_object = invariance(key);
        # [P1 P2 P3 O1 O2 O3 O4 O5]
        k = 9;


        property_vector = np.empty(int(n_props), bool);
        support_vector = np.empty(len(property_vector), float);
        
        property_vector[0] = invariance_object.P1();
        property_vector[1] = invariance_object.P2();
        property_vector[2] = invariance_object.P3();
        property_vector[3] = invariance_object.O1();
        property_vector[4] = invariance_object.O2();
        property_vector[5] = invariance_object.O3();
        property_vector[6] = invariance_object.O4();
        property_vector[7] = invariance_object.O5();
        property_vector[8] = invariance_object.uniform_scaling();
        

        for i,j in itertools.combinations(range(k),2):
            property_vector[k] = np.logical_and(property_vector[i],property_vector[j]);
            # print(str(k+1) + ' = ' + str(i) + ' and ' + str(j));
            k += 1;
            property_vector[k] = np.logical_or(property_vector[i],property_vector[j]);
            k += 1;

        # for i,j in itertools.combinations(range(7),2):
        #     property_vector[k] = np.logical_or(property_vector[i],property_vector[j]);
        #     # print(str(k) + ' = ' + str(i) + ' or ' + str(j));
        #     k += 1;

        # for i,j,l in itertools.combinations(range(7),3):
        #     property_vector[k] = np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]);
        #     k += 1;

        # for i,j,l,m in itertools.combinations(range(7),4):
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]);
        #     k += 1;

        # for i,j,l,m,n in itertools.combinations(range(7),5):
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_and(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
            
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_and(np.logical_or(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;

        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_and(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;

        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_and(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_or(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;
        #     property_vector[k] = np.logical_or(np.logical_or(np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]), property_vector[m]),property_vector[n]);
        #     k += 1;

        # properties_vector[key] = property_vector;
        properties_vector[value] = property_vector;

    return properties_vector;


def compute_property_vectors(measures_dict=measures_dict):

    # for single properties
    threshold_lower_1 = 0.00;
    threshold_upper_1 = 1;
    
    # for combinations of properties
    threshold_lower_2 = 0;
    threshold_upper_2 = 1;
    
    properties_vector = compute_property_vectors_without_support(measures_dict);
    property_names = np.array(inv);

    properties_vector_with_support_1 = np.empty((50,0));
    properties_names_with_support_1 = np.empty(0);
    properties_vector_with_support_2 = np.empty((50,0));
    properties_names_with_support_2 = np.empty(0);

    #nC1 properties
    (properties_vector_with_support_1, properties_names_with_support_1) = compute_property_vectors_with_support(properties_vector[:,0:9], property_names[0:9], threshold_lower_1, threshold_upper_2);

    #nC2 properties
    (properties_vector_with_support_2, properties_names_with_support_2) = compute_property_vectors_with_support(properties_vector[:,9:], property_names[9:], threshold_lower_2, threshold_upper_2);

    properties_vector_with_support = np.concatenate((properties_vector_with_support_1, properties_vector_with_support_2), axis=1);
    property_names = np.concatenate((properties_names_with_support_1, properties_names_with_support_2), axis=0);

    # print(properties_vector_with_support.shape);

    (support_array, entropy_array) = compute_supports(properties_vector_with_support);
    #convert property array to a dict
    # properties_dict = convert_property_array_to_dict(properties_vector_with_support, measures_dict);
    # convert zeros to -1s 
    # properties_vector_with_support = convert_zero_to_minus_one(properties_vector_with_support);

    return (properties_vector_with_support, property_names, support_array, entropy_array);

def entropy(vector):
    prob_vector = vector/np.sum(vector);
    n = len(vector);
    N = np.sum(vector);
    h = 0;
    for i in range(n):
        if(prob_vector[i]):
            h -= prob_vector[i] * np.log2(prob_vector[i]);

    return h/np.log2(n);

# def gini_index(vector):

    
def compute_homogeneity(property_vector, cluster_vector):
    n = len(property_vector);
    N_C = np.sum(cluster_vector);
    H = 0;

    for i in range(n):
        # number of measures possessing the property
        p1 = property_vector[i];
        # number of measures NOT possessing the property
        p2 = cluster_vector[i] - property_vector[i];
        
        #computing entropy for p1 and p2
        e = entropy(np.array([p1, p2]));
        
        #computing weighted average of the entropies, but not considering cluster of size 1 or below
        if (cluster_vector[i] > 1):
            H += e * cluster_vector[i] / N_C;

    return H;

def compute_property_frequencies_in_cluster_set(property_array, cluster_vector):
    (n_measures, n_properties) = property_array.shape;
    n_clusters = len(cluster_vector);
    cluster_property_vector = np.zeros((n_clusters, n_properties));

    for idx_clust, cluster in enumerate(cluster_vector):
        for idx_prop in range(n_properties):
            cluster_property_vector[idx_clust, idx_prop] = np.sum(property_array[cluster, idx_prop]);
    
    return np.transpose(cluster_property_vector);