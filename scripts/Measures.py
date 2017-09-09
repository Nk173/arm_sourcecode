# mapping measures with indices
import numpy as np;

n_measures = 50;



def initialize_measures_array():
    measures_arr = ['recall', 'precision', 'confidence', 'mutual_information', 'jaccard', 'f_measure', 'odds_ratio', 'specificity', 'negative_reliability', 'sebag_schoenauer', 'accuracy', 'support',
    'confidence_causal', 'lift', 'ganascia', 'kulczynsky_1', 'coverage',
    'prevalence', 'relative_risk', 'piatetsky_shapiro', 'novelty',
    'yules_q', 'yules_y', 'cosine', 'least_contradiction',
    'odd_multiplier', 'confirm_descriptive', 'confirm_causal',
    'certainty_factor', 'loevinger', 'conviction', 'information_gain',
    'laplace_correction', 'klosgen', 'zhang',
    'normalized_mutual_information', 'one_way_support',
    'two_way_support', 'implication_index', 'gini_index',
    'goodman_kruskal', 'leverage', 'kappa',
    'putative_causal_dependency', 'example_counterexample_rate',
    'confirmed_confidence_causal', 'added_value', 'collective_strength', 'j_measure', 'dependency'
    # 'theil_uncertainty_coefficient', 'TIC', 'logical_necessity',
    # 'kulczynsky_2', 'k_measure', 'interestingness_weighting_dependency', 'directed_information_ratio', 'chi_square', 'dilated_chi_square', 'conditional_entropy', 'complement_class_support', 'intensity_of_implication', 'correlation_coefficient'
    ];
    return np.array(measures_arr);


def map_measures_to_indices():
    measures_array = initialize_measures_array();
    measures_dict = form_measures_dict(measures_array);
    return (measures_dict, measures_array);

def form_measures_dict(measures_array):
    measures_dict = dict();
    for idx,measure in enumerate(measures_array):
        measures_dict[measure] = idx;
    return measures_dict;



    # measures['recall'] = 0;
    # measures['precision'] = 1;
    # measures['confidence'] = 2;
    # measures['mutual_information'] = 3;
    # measures['jaccard'] = 4;
    # measures['f_measure'] = 5;
    # measures['odds_ratio'] = 6;
    # measures['specificity'] = 7;
    # measures['negative_reliability'] = 8;
    # measures['sebag_schoenauer'] = 9;
    # measures['accuracy'] = 10;
    # measures['support'] = 11;
    # measures['confidence_causal'] = 12;
    # measures['lift'] = 13;
    # measures['ganascia'] = 14;
    # measures['kulczynsky_1'] = 15;
    # measures['coverage'] = 16;
    # measures['prevalence'] = 17;
    # measures['relative_risk'] = 18;
    # measures['piatetsky_shapiro'] = 19;
    # measures['novelty'] = 20;
    # measures['yules_q'] = 21;
    # measures['yules_y'] = 22;
    # measures['cosine'] = 23;
    # measures['least_contradiction'] = 24;
    # measures['odd_multiplier'] = 25;
    # measures['confirm_descriptive'] = 26;
    # measures['confirm_causal'] = 27;
    # measures['certainty_factor'] = 28;
    # measures['loevinger'] = 29;
    # measures['conviction'] = 30;
    # measures['information_gain'] = 31;
    # measures['laplace_correction'] = 32;
    # measures['klosgen'] = 33;
    # measures['zhang'] = 34;
    # measures['normalized_mutual_information'] = 35;
    # measures['one_way_support'] = 36;
    # measures['two_way_support'] = 37;
    # measures['implication_index'] = 38;
    # measures['gini_index'] = 39;
    # measures['goodman_kruskal'] = 40;
    # measures['leverage'] = 41;
    # measures['kappa'] = 42;
    # measures['putative_causal_dependency'] = 43;
    # measures['example_counterexample_rate'] = 44;
    # measures['confirmed_confidence_causal'] = 45;
    # measures['added_value'] = 46;
    # measures['collective_strength'] = 47;
    # measures['j_measure'] = 48;
    # measures['dependency'] = 49;