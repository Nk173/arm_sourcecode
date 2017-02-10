# mapping measures with indices
import numpy as np;

n_measures = 50;

def map_measures_to_indices():
    measures = dict();
    measures['recall'] = 0;
    measures['precision'] = 1;
    measures['confidence'] = 2;
    measures['mutual_information'] = 3;
    measures['jaccard'] = 4;
    measures['f_measure'] = 5;
    measures['odds_ratio'] = 6;
    measures['specificity'] = 7;
    measures['negative_reliability'] = 8;
    measures['sebag_schoenauer'] = 9;
    measures['accuracy'] = 10;
    measures['support'] = 11;
    measures['confidence_causal'] = 12;
    measures['lift'] = 13;
    measures['ganascia'] = 14;
    measures['kulczynsky_1'] = 15;
    measures['coverage'] = 16;
    measures['prevalence'] = 17;
    measures['relative_risk'] = 18;
    measures['piatetsky_shapiro'] = 19;
    measures['novelty'] = 20;
    measures['yules_q'] = 21;
    measures['yules_y'] = 22;
    measures['cosine'] = 23;
    measures['least_contradiction'] = 24;
    measures['odd_multiplier'] = 25;
    measures['confirm_descriptive'] = 26;
    measures['confirm_causal'] = 27;
    measures['certainty_factor'] = 28;
    measures['loevinger'] = 29;
    measures['conviction'] = 30;
    measures['information_gain'] = 31;
    measures['laplace_correction'] = 32;
    measures['klosgen'] = 33;
    measures['zhang'] = 34;
    measures['normalized_mutual_information'] = 35;
    measures['one_way_support'] = 36;
    measures['two_way_support'] = 37;
    measures['implication_index'] = 38;
    measures['gini_index'] = 39;
    measures['goodman_kruskal'] = 40;
    measures['leverage'] = 41;
    measures['kappa'] = 42;
    measures['putative_causal_dependency'] = 43;
    measures['example_counterexample_rate'] = 44;
    measures['confirmed_confidence_causal'] = 45;
    measures['added_value'] = 46;
    measures['collective_strength'] = 47;
    measures['j_measure'] = 48;
    measures['dependency'] = 49;


    #mapping of indices to measure names
    measures_arr = np.empty(n_measures, object);
    for key,value in measures.items():
    #     print(value)
        measures_arr[value] = str(key);
        
    return (measures, measures_arr);