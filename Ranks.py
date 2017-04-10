import numpy as np;
from scipy.stats import spearmanr;
from scipy.spatial.distance import pdist, squareform;
from scipy.cluster import hierarchy;
import matplotlib.pyplot as plt;

from Measures import map_measures_to_indices, form_measures_dict;

(measures_dict, measures_arr) = map_measures_to_indices();

class ranks(object):
    def __init__ (self, scores_matrix, measures_array = measures_arr, measures_dictionary = measures_dict):
        self.scores = scores_matrix;
        self.ranks = np.zeros(shape=scores_matrix.shape);
        self.compute_ranks();
        self.measures_arr = measures_array;
        self.measures_dict = measures_dictionary;
    
    def compute_ranks(self):
        for idx,score in enumerate(self.scores.T):
            self.ranks[:,idx] = self.return_ranks(score);
    
    def return_ranks(self, scores_array):
        
        # takes mxn
        temp = np.argsort(scores_array)[::-1];

        ranks_array = np.empty(len(scores_array), float);
        
        #   Assigning ranks to the scores according to the order in 'temp' array (descending score)
        ranks_array[temp] = np.arange(len(scores_array)) + 1;
        
        scores_array = np.around(scores_array, decimals=8);
        unique, counts = np.unique(scores_array, return_counts=True);

        # new version - nan scores get lowest values
        #   Assigning nan ranks to nan scores
        # ranks_array[np.isnan(scores_array)] = np.nan;

        # unique, counts = np.unique(scores_array[~np.isnan(scores_array)], return_counts=True);

        # Handle ties by assigning averaged values to the tied scores
        for idx,u in enumerate(unique):
            ranks_array[scores_array==u] = ranks_array[scores_array==u].sum()/counts[idx];
        return ranks_array;
    
    def visualize(self, linkage_method='complete'):
        self.compute_correlation();
        self.show_correlation_matrix();
        self.compute_distance();
        self.show_dendrogram(linkage_method);
    
    def compute_correlation(self):
        self.corr_spearman = spearmanr(self.ranks).correlation;
        
    def compute_distance(self):
#         if not hasattr (self, 'corr_spearman'):
#             self.compute_correlation();
        self.compute_correlation();
        # self.distance = -self.corr_spearman;
        self.distance = np.sqrt((1 - self.corr_spearman) / 2);
        # self.distance = (1 - self.corr_spearman) / 2;
        self.distance_1D = squareform(self.distance, checks=False);
        self.distance_uppertriangle = squareform(self.distance_1D);

    def compute_clusters(self, method='complete'):
#         if not hasattr (self, 'distance_uppertriangle'):
#             self.compute_distance();
        self.compute_distance();
        self.cluster = hierarchy.linkage(self.distance_uppertriangle, method=method);
        
    def form_clusters(self, n_clusters, method='complete'):
#         if not hasattr(self, 'cluster'):
#             self.compute_clusters();
        self.compute_clusters(method);
        clusters = hierarchy.cut_tree(self.cluster, n_clusters=n_clusters);
        clusters_arr = np.empty(n_clusters,object);
        for n in range(n_clusters):
            clusters_arr[n] = (clusters==n).nonzero()[0];
        return clusters_arr;
        
    def show_correlation_matrix(self):
        if not hasattr (self, 'corr_spearman'):
            self.compute_correlation();
            
        fig, ax = plt.subplots(figsize=[10,8], ncols=1, nrows=1);
        # plt.pcolor(self.corr_spearman, cmap = 'RdBu');
        plt.imshow(self.corr_spearman, interpolation='none', cmap= 'RdBu');
        # plt.axis([0,self.corr_spearman.shape[1],0,self.corr_spearman.shape[0]]);
        
        ax.set_xticks(np.arange(len(self.measures_arr)), minor=False);
        ax.set_xticklabels(self.measures_arr, minor=False);
        plt.xticks(rotation=90);
        
        ax.set_yticks(np.arange(len(self.measures_arr)), minor=False);
        ax.set_yticklabels(self.measures_arr, minor=False);

        plt.colorbar();
        plt.show();

    def show_dendrogram(self, linkage_method='complete'):
        # if not hasattr(self, 'cluster'):
        #     self.compute_clusters();
        self.compute_clusters(method=linkage_method);            
        plt.figure(figsize=(10,10));
        
        def llf(id):
            return self.measures_arr[id];

        a = hierarchy.dendrogram(self.cluster, orientation='right', 
                                    color_threshold=1, leaf_label_func= llf,
                                    leaf_font_size = 7);
        plt.show();
        
    def remove_outliers(self, indices):
        self.ranks = np.delete(arr=self.ranks, obj=indices, axis=1);
        self.measures_arr = np.delete(arr=self.measures_arr, obj=indices, axis=0);
        self.measures_dict = form_measures_dict(self.measures_arr);
        self.compute_correlation();
        return (self.measures_dict, self.measures_arr);