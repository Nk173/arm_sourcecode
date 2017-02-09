import numpy as np;
from scipy.stats import spearmanr;
from scipy.spatial.distance import pdist, squareform;
from scipy.cluster import hierarchy;
import matplotlib.pyplot as plt;


class ranks(object):
    def __init__ (self, scores_matrix, measures_arr):
        self.scores = scores_matrix;
        self.ranks = np.zeros(shape=scores_matrix.shape);
        self.compute_ranks();
        self.measures_arr = np.array(measures_arr);
    
    def compute_ranks(self):
        for idx,score in enumerate(self.scores.T):
            self.ranks[:,idx] = self.return_ranks(score);
    
    def return_ranks(self, scores_array):
        # takes mxn
        temp = np.argsort(scores_array)[::-1];

        ranks_array = np.empty(len(scores_array), float);
        #   Assigning ranks to the scores according to the order in 'temp' array (descending score)
        ranks_array[temp] = np.arange(len(scores_array));
        #   Assigning nan ranks to nan scores
        ranks_array[np.isnan(scores_array)] = np.nan;

        unique, counts = np.unique(scores_array[~np.isnan(scores_array)], return_counts=True);

        # Handle ties by assigning averaged values to the tied scores
        for idx,u in enumerate(unique):
            ranks_array[scores_array==u] = ranks_array[scores_array==u].sum()/counts[idx];
        return ranks_array;
    
    def visualize(self):
        self.compute_correlation();
        self.show_correlation_matrix();
        self.compute_distance();
        self.show_dendrogram();
    
    def compute_correlation(self):
        self.corr_spearman = spearmanr(self.ranks).correlation;
        
    def compute_distance(self):
#         if not hasattr (self, 'corr_spearman'):
#             self.compute_correlation();
        self.compute_correlation();
        self.distance = np.sqrt((1 - self.corr_spearman) / 2);
        self.distance_1D = squareform(self.distance, checks=False);
        self.distance_uppertriangle = squareform(self.distance_1D);

    def compute_clusters(self):
#         if not hasattr (self, 'distance_uppertriangle'):
#             self.compute_distance();
        self.compute_distance();
        self.cluster = hierarchy.linkage(self.distance_uppertriangle, method='complete');
        
    def form_clusters(self, n_clusters):
#         if not hasattr(self, 'cluster'):
#             self.compute_clusters();
        self.compute_clusters();
        clusters = hierarchy.cut_tree(self.cluster, n_clusters=n_clusters);
        for n in range(n_clusters):
            print(n,np.count_nonzero((clusters==n)), (clusters==n).nonzero()[0]);
        
    def show_correlation_matrix(self):
        if not hasattr (self, 'corr_spearman'):
            self.compute_correlation();
            
        fig, ax = plt.subplots(figsize=[10,10], ncols=1, nrows=1);
        plt.pcolor(self.corr_spearman, cmap = 'RdBu');
        plt.axis([0,self.corr_spearman.shape[1],0,self.corr_spearman.shape[0]]);
        plt.colorbar();
        plt.show();

    def show_dendrogram(self):
        if not hasattr(self, 'cluster'):
            self.compute_clusters();
        plt.figure(figsize=(10,10));
        hierarchy.dendrogram(self.cluster, orientation='right', color_threshold=1);
        plt.show();
        
    def remove_outliers(self, indices):
        self.ranks = np.delete(arr=self.ranks, obj=indices, axis=1);
        self.measures_arr = np.delete(arr=self.measures_arr, obj=indices, axis=0);
        self.compute_correlation()