import numpy as np;
from Invariance import invariance;
from Measures import map_measures_to_indices;

(measures_dict, measures_arr) = map_measures_to_indices();
properties_vector = dict();

# Can take a dict of measure names as keys if needed, else will pull the original one in Measures module
# returns a dict with property vectors as values
def compute_property_vectors(measures_dict=measures_dict):
    for key,value in measures_dict.items():
        invariance_object = invariance(key);
        # [P2 P3 O1 O2 O3 O4 O5]
        property_vector = np.empty(7, bool);
        property_vector[0] = invariance_object.P2();
        property_vector[1] = invariance_object.P3();
        property_vector[2] = invariance_object.O1();
        property_vector[3] = invariance_object.O2();
        property_vector[4] = invariance_object.O3();
        property_vector[5] = invariance_object.O4();
        property_vector[6] = invariance_object.O5();
        properties_vector[key] = property_vector;
    return properties_vector;