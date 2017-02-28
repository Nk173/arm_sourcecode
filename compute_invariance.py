import numpy as np;
from Invariance import invariance;
from Measures import map_measures_to_indices;
import itertools;
import scipy.misc as sm;

(measures_dict, measures_arr) = map_measures_to_indices();

#Converts the 0/1 binary array to -1/+1 array
def convert_zero_to_minus_one(properties_vector):
    
    properties_vector_modified = dict();
    for key, value in properties_vector.items():
        property_vector = np.empty(len(value), int);
        for idx, element in enumerate(value):
            if element:
                property_vector[idx] = 1;
            else:
                property_vector[idx] = -1;                

        properties_vector_modified[key] = property_vector;

    return properties_vector_modified


# Can take a dict of measure names as keys if needed, else will pull the original one in Measures module
# returns a dict with property vectors as values
def compute_property_vectors(measures_dict=measures_dict):
    properties_vector = dict();
    for key,value in measures_dict.items():
        invariance_object = invariance(key);
        # [P2 P3 O1 O2 O3 O4 O5]

        n_props = 7 + sm.comb(7,2) + sm.comb(7,2) + sm.comb(7,3) + sm.comb(7,3);

        property_vector = np.empty(int(n_props), bool);

        property_vector[0] = invariance_object.P2();
        property_vector[1] = invariance_object.P3();
        property_vector[2] = invariance_object.O1();
        property_vector[3] = invariance_object.O2();
        property_vector[4] = invariance_object.O3();
        property_vector[5] = invariance_object.O4();
        property_vector[6] = invariance_object.O5();

        k = 7
        for i,j in itertools.combinations(range(7),2):
            property_vector[k] = np.logical_and(property_vector[i],property_vector[j]);
            # print(str(k) + ' = ' + str(i) + ' and ' + str(j));
            k += 1;
        
        for i,j in itertools.combinations(range(7),2):
            property_vector[k] = np.logical_or(property_vector[i],property_vector[j]);
            # print(str(k) + ' = ' + str(i) + ' or ' + str(j));
            k += 1;

        for i,j,l in itertools.combinations(range(7),3):
            property_vector[k] = np.logical_and(np.logical_and(property_vector[i],property_vector[j]), property_vector[l]);
            # print(str(k) + ' = ' + str(i) + ' or ' + str(j));
            k += 1;

        for i,j,l in itertools.combinations(range(7),3):
            property_vector[k] = np.logical_or(np.logical_or(property_vector[i],property_vector[j]), property_vector[l]);
            # print(str(k) + ' = ' + str(i) + ' or ' + str(j));
            k += 1;


        properties_vector[key] = property_vector;

        properties_vector_modified = convert_zero_to_minus_one(properties_vector);
    return properties_vector_modified;