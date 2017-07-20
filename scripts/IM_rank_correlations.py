import numpy as np;
from Contingency_Table import contingency_table, generate_contingency_tables;
from Measures import map_measures_to_indices;
from Ranks import ranks;



def create_ranks_matrix(tables_matrix, measures_arr):
    n_measures = len(measures_arr);
    scores_matrix = np.zeros(shape=(len(tables_matrix),n_measures));

    for idx,table in enumerate(tables_matrix):
        t = contingency_table(table);
        t.compute_scores();
        scores_matrix[idx] = t.scores;

    ranks_matrix = ranks(scores_matrix, measures_arr);
    return (ranks_matrix, scores_matrix)

# perturbing the values away from multiples of 10 to avoid inadvertent factorization
#dense condition
# vals_dense 
vals_dense = np.array([
    [5425, 4482, 6880, 1237, 3154, 3506, 3740],
    [392, 620, 933, 671, 418, 748, 939],
    [116, 796, 721, 457, 12, 224, 19],
    [0,1,10,11]]);

vals_sparse = np.array([
    [0,1,10,11],
    [573, 664, 536,  16, 612, 815, 410],
    [573, 664, 536,  16, 612, 815, 410],
    [57013, 21664, 5136,  1106, 36112, 18115, 47110]]);

#mix condition
vals_mix = np.array([
    [0, 5732, 5318, 3090, 765, 4757, 5081, 615],
    [0, 1198, 3255, 4784, 877, 7444, 8413, 4818],
    [0, 601, 2158, 3446, 1706, 4000, 5484, 2396],
    [0, 7067, 4117, 458, 5717, 9940, 2814, 6672]]);

#generates contingency tables with all f_ij's taking all possibile values in 'vals' array
tables_dense = generate_contingency_tables(vals_dense);
tables_sparse = generate_contingency_tables(vals_sparse);
tables_mix = generate_contingency_tables(vals_mix);

# maps measures to indices and outputs a dictionary and an array
(measures_dict, measures_arr) = map_measures_to_indices();

#computes the ranks class with the given scores
(ranks_matrix_dense, scores_matrix_dense) = create_ranks_matrix(tables_dense, measures_arr);
# ranks_matrix_dense = ranks(scores_matrix_dense, measures_arr);
(ranks_matrix_sparse, scores_matrix_sparse) = create_ranks_matrix(tables_sparse, measures_arr);
# ranks_matrix_sparse = ranks(scores_matrix_sparse, measures_arr);
(ranks_matrix_mix, scores_matrix_mix) = create_ranks_matrix(tables_mix, measures_arr);
# ranks_matrix_mix = ranks(scores_matrix_mix, measures_arr);