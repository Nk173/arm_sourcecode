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

inv = np.array(inv);


def new_property_vectors():
    props = dict();
    prop_names = ['UNAI_f11','UNAI_f00','UNAI_f10','UNAI_f01','UNAI','UNZR_f11','UNZR_f00','UNZR_f10','UNZR_f01','UNZR'];
    props['lift'] = [1,-1,1,1,-1,1,0,0,0,0];
    props['jaccard'] = [1,1,1,1,1,1,-1,0,0,-1];
    props['confidence'] = [1,1,1,1,1,1,-1,1,-1,-1];
    props['precision'] = [1,1,1,1,1,1,-1,1,-1,-1];
    props['recall'] = [1,1,1,1,1,1,-1,-1,1,-1];
    props['specificity'] = [1,1,1,1,1,-1,1,-1,1,-1];
    props['ganascia'] = [1,1,1,1,1,1,-1,1,-1,-1];
    props['kulczynsky_1'] = [-1,1,1,1,-1,1,-1,0,0,-1];
    props['f_measure'] = [1,1,1,1,1,1,-1,0,0,-1];
    props['confidence_causal'] = [1,1,1,1,1,1,1,1,-1,-1];
    props['odds_ratio'] = [-1,-1,1,1,-1,0,0,0,0,0];
    props['negative_reliability'] = [1,1,1,1,1,-1,1,-1,1,-1];
    props['sebag_schoenauer'] = [-1,1,1,1,-1,1,-1,0,-1,-1]
    props['accuracy'] = [1,1,1,1,1,0,0,0,0,0];
    props['support'] = [1,1,1,1,1,1,-1,0,0,-1];
    props['coverage'] = [1,1,1,1,1,0,-1,-1,0,-1];
    props['prevalence'] = [1,1,1,1,1,0,-1,0,-1,-1];
    props['relative_risk'] = [1,-1,1,1,-1,1,0,1,0,0];
    props['novelty'] = [1,1,1,1,1,1,1,1,1,1];
    props['yules_q'] = [1,1,1,1,1,0,0,0,0,0];
    props['yules_y'] = [1,1,1,1,1,0,0,0,0,0];
    props['cosine'] = [1,1,1,1,1,1,-1,1,1,-1];
    props['least_contradiction'] = [1,1,-1,1,-1,1,-1,1,-1,-1];
    props['odd_multiplier'] = [1,-1,1,1,-1,1,0,0,1,0];
    props['confirm_descriptive'] = [1,1,1,1,1,1,-1,1,-1,-1];
    props['confirm_causal'] = [1,1,1,1,1,1,1,1,-1,-1];
    props['certainty_factor'] = [1,1,1,-1,-1,0,0,-1,1,-1];
    props['loevinger'] = [1,1,1,-1,-1,0,0,-1,1,-1];
    props['conviction'] = [1,1,1,1,1,0,0,0,1,0];
    props['information_gain'] = [1,1,1,1,1,1,1,0,0,0];
    props['laplace_correction'] = [1,1,1,1,1,1,-1,1,-1,-1];
    props['klosgen'] = [1,1,1,1,1,0,-1,-1,-1,-1];
    props['piatetsky_shapiro'] = [1,1,1,1,1,1,1,1,1,1];
    props['zhang'] = [1,-1,1,-1,-1,1,0,1,0,0];
    props['one_way_support'] = [1,1,1,1,1,-1,0,-1,0,-1];

    props['two_way_support'] = [1,1,1,1,1,-1,0,1,1,-1];
    props['implication_index'] = [1,1,1,1,1,-1,-1,-1,-1,-1];
    props['leverage'] = [1,1,1,1,1,1,0,1,-1,-1];
    props['kappa'] = [1,1,1,1,1,0,0,1,1,0];
    props['confirmed_confidence_causal'] = [1,1,1,1,1,1,1,1,-1,-1];
    props['example_counterexample_rate'] = [1,1,-1,1,-1,0,-1,1,-1,-1];
    props['putative_causal_dependency'] = [1,1,1,1,1,0,0,1,1,0];
    props['dependency'] = [1,1,1,1,1,0,0,0,0,0];
    props['j_measure'] = [1,1,1,1,1,-1,-1,1,-1,-1];
    props['collective_strength'] = [1,1,1,1,1,1,1,1,1,1];
    props['gini_index'] = [1,1,1,1,1,-1,-1,0,0,-1];
    props['goodman_kruskal'] = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
    props['mutual_information'] = [1,1,1,1,1,-1,-1,1,1,-1];
    props['normalized_mutual_information'] = [1,1,1,1,1,-1,-1,-1,-1,-1];
    props['added_value'] = [-1,1,-1,-1,-1,0,0,0,0,0];

    return (props, prop_names);

#checks whether the values are a specific combination of n/p/y respectively
def UN_combination_value(value1, value2):
    if (value1 == -1) and (value2 == -1):
            return 1;
    elif (value1 == -1) and (value2 == 0):
            return 2;
    elif (value1 == -1) and (value2 == 1):
            return 3;
    elif (value1 == 1) and (value2 == -1):
            return 4;
    elif (value1 == 1) and (value2 == 0):
            return 5;
    elif (value1 == 1) and (value2 == 1):
            return 6;
    return 0;

# takes a property array and returns support array and corresponding measure of impurity
def compute_supports(properties_vector):
    (n_measures, n_props) = properties_vector.shape;    
    
    support_array = np.empty(n_props, float);
    entropy_array = np.empty(n_props, float);

    for i in range(n_props):
        prop_vector = properties_vector[:,i];

        n = np.sum(properties_vector[:,i]);
        N = len(properties_vector[:,i]);
        support_array[i] =  n/N ;
        entropy_array[i] = entropy(np.array([n, N-n]));

    return (support_array, entropy_array);

def compute_supports_updated(properties_vector):
    (n_measures, n_props) = properties_vector.shape;    
    u_states = np.unique(properties_vector);

    support_array = np.empty((n_props,len(u_states)), float);
    entropy_array = np.empty((n_props), float);

    for i in range(n_props):
        prop_vector = properties_vector[:,i];

        N = len(prop_vector);
        
        u_vect = np.unique(prop_vector);
        counts = np.zeros(len(u_states));
        
        for idx_u in range(len(u_states)):
            for p_m in prop_vector:
                if (p_m == u_states[idx_u]):
                    counts[idx_u] += 1;
                    
        support_array[i,:] = counts/N;
        entropy_array[i] = entropy(counts);
#         n = np.sum(properties_vector[:,i]);
#         N = len(properties_vector[:,i]);
#         support_array[i] =  n/N ;
#         entropy_array[i] = entropy(np.array([n, N-n]));

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


def compute_property_vectors(measures_dict=measures_dict, property_names=inv):

    # for single properties
    threshold_lower_1 = 0.00;
    threshold_upper_1 = 1;
    
    # for combinations of properties
    threshold_lower_2 = 0;
    threshold_upper_2 = 1;
    
    properties_vector = compute_property_vectors_without_support(measures_dict);
    # property_names = np.array(inv);

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

    (support_array, entropy_array) = compute_supports_updated(properties_vector_with_support);
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
        if(prob_vector[i]<0):
            print('frequency lesser than 0', vector);
        if(prob_vector[i]>0):
            h -= prob_vector[i] * np.log2(prob_vector[i]);

    return h/np.log2(n);

# def gini_index(vector):

    
def compute_homogeneity(property_vector, cluster_vector):
    # property_vector - n_prop_states x n_clusters
    # cluster_vector - n_clusters x 1

    # number of clusters
    n = len(cluster_vector);
    N_C = np.sum(cluster_vector);
    H = 0;

    for i in range(n):
        e = entropy(property_vector[:,i]);
        # number of measures possessing the property
        # p1 = property_vector[i];
        # number of measures NOT possessing the property
        # p2 = cluster_vector[i] - property_vector[i];
        
        #computing entropy for p1 and p2
        # e = entropy(np.array([p1, p2]));
        
        #computing weighted average of the entropies, but not considering cluster of size 1 or below
        if (cluster_vector[i] > 1):
            H += e * cluster_vector[i] / N_C;

    return H;

# when there are 2 states of property status
def compute_property_frequencies_in_cluster_set(property_array, cluster_vector):
    (n_measures, n_properties) = property_array.shape;
    n_clusters = len(cluster_vector);
    cluster_property_vector = np.zeros((n_clusters, n_properties));

    for idx_clust, cluster in enumerate(cluster_vector):
        for idx_prop in range(n_properties):
            cluster_property_vector[idx_clust, idx_prop] = np.sum(property_array[cluster, idx_prop]);
    
    return np.transpose(cluster_property_vector);

def map_UN_properties_to_combinations(property_array):
    (n_measures, n_props_old) = property_array.shape;
    n_props_new = 16;
    new_property_array = np.empty((n_measures, n_props_new), int);
    new_property_names = np.empty(n_props_new, object);
    
    freq_array = ['f11', 'f10', 'f01', 'f00'];
    k = 0;
        # UNAI_fij
    for i in range(len(freq_array)):
        # UNZR_fij
        for j in range(len(freq_array)):
            new_property_names[k] = freq_array[i] + '_' + freq_array[j];
            for m in range(n_measures):
                p_vector = property_array[m,:];
                new_property_array[m,k] = UN_combination_value(p_vector[i],p_vector[j+5]);
            k += 1;
    return (new_property_array, new_property_names);



# can take any number of property states and returns a 
# cluster property vector of dimensions
# (n_props, n_prop_states, n_clusters)
def compute_property_frequencies_in_cluster_set_updated(property_array, cluster_vector):
    (n_measures, n_properties) = property_array.shape;
    n_clusters = len(cluster_vector);
    u_states = np.unique(property_array);
#     print(u_states)
    cluster_property_vector = np.zeros((n_properties, len(u_states), n_clusters));

    for idx_clust, cluster in enumerate(cluster_vector):
        for idx_prop in range(n_properties):
            count = np.zeros(len(u_states));
            cluster_prop_vector = property_array[cluster, idx_prop];
            for i in range(len(u_states)):
                for prop_val in cluster_prop_vector:
                    if (prop_val == u_states[i]):
                        count[i] += 1;
            cluster_property_vector[idx_prop,:,idx_clust] = count
    return cluster_property_vector;

# returns the property vector - (n_measures, n_props)
def compute_new_property_vectors(measures_dict):
    # initialize default properties dict
    (prop, prop_names) = new_property_vectors();
    n_props = len(prop_names);
    properties_array = np.empty((len(measures_dict),n_props), int);
    for key,value in measures_dict.items():
        properties_array[value,:] = prop[key];
    
    # k = n_props;
    # for i,j in itertools.combinations(range(k),2):
    #     properties_array[k] = np.logical_and(properties_array[i],properties_array[j]);
    #     # print(str(k+1) + ' = ' + str(i) + ' and ' + str(j));
    #     k += 1;
    #     properties_array[k] = np.logical_or(properties_array[i],properties_array[j]);
    #     k += 1;


    return (properties_array, prop_names);