import numpy as np;
from Invariance_updated import invariance_updated;
from Measures import map_measures_to_indices;
import itertools;

(measures_dict, measures_arr) = map_measures_to_indices();

# Can take a dict of measure names as keys if needed, else will pull the original one in Measures module
# returns a dict with property vectors as values
def compute_property_vectors(measures_dict=measures_dict):
    properties_vector = dict();
    for key,value in measures_dict.items():
        invariance_object = invariance_updated(key);

        property_vector = np.empty(10, bool);

        property_vector[0] = invariance_object.P1();
        property_vector[1] = invariance_object.P2();
        property_vector[2] = invariance_object.P3();
        property_vector[3] = invariance_object.P4();
        property_vector[4] = invariance_object.P5();
        property_vector[5] = invariance_object.P6();
        property_vector[6] = invariance_object.P7();
        property_vector[7] = invariance_object.P8();
        property_vector[8] = invariance_object.P9();
        property_vector[9] = invariance_object.P9();
        
        
        # print(key, property_vector[0], property_vector[1], property_vector[2]);
        properties_vector[key] = property_vector;
    return properties_vector;