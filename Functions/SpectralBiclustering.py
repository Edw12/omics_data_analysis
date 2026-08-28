# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:25:29 2026

@author: Edwin
"""

from sklearn.cluster import SpectralBiclustering
from sklearn.preprocessing import scale
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score

from statsmodels.stats.multitest import multipletests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb

from scipy import stats
from scipy.cluster.hierarchy import fcluster

from load_human_metabolites import load_human_metabolites

# Source - https://stackoverflow.com/a/14463362
# Posted by Mike, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-07, License - CC BY-SA 4.0

import warnings
warnings.filterwarnings("ignore")

# Suppress all warnings


    
class Spectral_Biclustering:
    """
    The general pipeline for using this class is to initialise the data, perform the clustering, generate the purity test
    and then use the clustering groupings. The purity test tells you which row clusters (which correspond to clustered datapoints)
    correspond to which true groups (sometimes there will be 1 true group, sometimes multiple). The clustering groupings
    dataframe then tells you which column clusters are associatted with each sample clusters. The column clusters contain a set
    of metabolites that could be linked to the group selection. Because there is no validation/testing and the datasets
    used are often quite small, conclusions can only be drawn very weakly. It is hypothesis-generating not hypothesis-
    confirming.
    """
    def __init__(self, data_func, data_frame = False, random_state = 42):
        """
        Takes a function that loads the data for the clustering. If data_frame = True, assumes that the data is returned as
        a data frame with the groups in a seperate column called group.

        Parameters
        ----------
        data_func : function
            Should load the data, in the form x = pandas df of data, y = numpy array of true groups, extra outputs. Alternatively,
            loads a single pandas dataframe, with a column called "Group" containing the true groups

        Returns
        -------
        Saves the necissary data
        """
        if data_frame == True:
            data = data_func()
            x = data.drop("Group", axis = 1)
            y = np.array(data["Group"].values)
        else:
            x, y, *_ = data_func()

        x_numeric = x.apply(pd.to_numeric, errors='coerce')  # Makes sure not saved as strings
        
        self.x = x
        self.x_numeric = x_numeric
        self.y = y
        self.random_state = random_state
    
    def perform_cluster(self, n_row_clusters = 6, n_col_clusters = 9):
        """
        Performs the Spectral Biclustering and prints the adjusted rand score, the normalised mutual information
        score and the adjusted mutual information score. It then prints the p value, testing whether the 
        model performs better than random.
        
        Parameters
        ----------
        n_row_clusters : int, optional
            Number of row clusters. The default is 6.
        group_b : int, optional
            Number of col clusters. The default is 9.
            
        Returns
        -------
        Must be performed before any analysis can be done on the clustering
        """
        # X: shape (30, 600), rows = samples, cols = metabolites
        # Preprocess — this matters more than the algorithm here
        X_log = np.log1p(self.x_numeric)                      # log-transform (metabolomics is usually right-skewed)
        X_scaled = scale(X_log, axis=0)          
        
        self.X_scaled = X_scaled
        
        # Fit
        self.n_row_clusters = n_row_clusters   # your hypothesized number of sample subgroups
        self.n_col_clusters = n_col_clusters   # number of metabolite modules — look at silhouette score
    
        model = SpectralBiclustering(
            n_clusters=(self.n_row_clusters, self.n_col_clusters),
            method='log',        # 'log' works well for count/intensity-like data; 'bistochastic' is the alternative
            random_state=self.random_state
            )
        model.fit(X_scaled)

        # Extract results
        row_labels = model.row_labels_       # cluster assignment per sample 
        col_labels = model.column_labels_    # cluster assignment per metabolite 
        
        self.row_labels = row_labels
        self.col_labels = col_labels 
        
        # Reorder the matrix to visualize the checkerboard structure
        fit_order = np.argsort(row_labels)
        col_order = np.argsort(col_labels)
        self.X_reordered = X_scaled[fit_order][:, col_order]

        
        # Comparing the clusters with the true labels

        ari = adjusted_rand_score(self.y, row_labels)         # 1.0 is perfect match, 0.0 is effectively random <- whether pairs in one group are put in the same group
        nmi = normalized_mutual_info_score(self.y, row_labels)  # "" <- whether knowing the clusters makes knowing the true groups easier
        ami = adjusted_mutual_info_score(self.y, row_labels) # "" <- chance corrected overlap comparisons

        print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}, AMI: {ami:.3f}")

        # Shuffle the true labels 1000 times and generates an ari score, checks if any are better than the modelled version

        null_aris = []
        for i in range(1000):
            y_shuffled = shuffle(self.y, random_state=i)
            null_aris.append(adjusted_rand_score(y_shuffled, row_labels))
        
        p_value = np.mean(np.array(null_aris) >= ari)

        print(f"The p value is : {p_value}")
        
    
    def plot_biclusters(self, cmap="RdBu_r", 
                        vmin=None, vmax=None, figsize=(12, 8)):
        """
        Plots the results of the Spectral Biclustering algorithm. Unlike the Seaborn plot, shows you the
        clusters from the clustering algorithm actually performed

        Parameters
        ----------
        cmap : string, matplotlib colourmap, optional
            The colourmap to be used. The default is "RdBu_r".
        vmin : int, optional
            vmin for the imshow call. The default is None.
        vmax : int, optional
            vmax for the imshow call. The default is None.
        figsize : tuple of int, optional
            The size of the figure. The default is (12, 8).
        """
        
        fit_order = np.argsort(self.row_labels)  # Finds the order for the rows and columns
        col_order = np.argsort(self.col_labels)
    
        sorted_row_labels = self.row_labels[fit_order]
        sorted_col_labels = self.col_labels[col_order]
    
        # symmetric color scale is usually best for z-scored data
        if vmin is None or vmax is None:
            v = np.nanmax(np.abs(self.X_reordered))
            vmin, vmax = -v, v
    
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.X_reordered, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)  # Heatmap of the clusters
    
        # --- draw lines at cluster boundaries ---
        row_boundaries = np.where(np.diff(sorted_row_labels) != 0)[0] + 0.5
        col_boundaries = np.where(np.diff(sorted_col_labels) != 0)[0] + 0.5
    
        for b in row_boundaries:
            ax.axhline(b, color="black", linewidth=1)
        for b in col_boundaries:
            ax.axvline(b, color="black", linewidth=1)
    
        # --- label each cluster block at its center ---
        row_starts = np.concatenate(([0], row_boundaries, [self.X_reordered.shape[0]]))
        col_starts = np.concatenate(([0], col_boundaries, [self.X_reordered.shape[1]]))
    
        row_centers = (row_starts[:-1] + row_starts[1:]) / 2
        col_centers = (col_starts[:-1] + col_starts[1:]) / 2
    
        unique_row_clusters = np.unique(sorted_row_labels)
        unique_col_clusters = np.unique(sorted_col_labels)
    
        ax.set_yticks(row_centers)
        ax.set_yticklabels([f"Row cluster {c}" for c in unique_row_clusters])
    
        ax.set_xticks(col_centers)
        ax.set_xticklabels([f"Col cluster {c}" for c in unique_col_clusters], rotation=90)
    
        ax.set_xlabel("Metabolites (reordered)")
        ax.set_ylabel("Samples (reordered)")
        ax.set_title("Spectral Biclustering — reordered data matrix")
    
        fig.colorbar(im, ax=ax, label="Scaled log-intensity")
        plt.tight_layout()
        plt.show()
    
        return fig, ax
    
    def cluster_groupings(self, show = True, output = False):
        """
        Returns the metabolites in a cluster from the biclustering. Also returns a table of the mean
        z value for each cluster with each true group to see which clusters of metabolites elevate
        or reduce a group's selection.
        
        Params
        ------
        show : bool, optional
        If True, prints the outputs. Default is True
        output : bool, optional
        If True, returns information as a list. Default is False
        Returns
        -------
        Must have run init and perform_cluster first
        """
        outputs = []
        for c in range(self.n_col_clusters):
            metabolites_in_cluster = self.x.columns[self.col_labels == c]
            if show == True:
                print(f"Cluster {c} (n={len(metabolites_in_cluster)}): {list(metabolites_in_cluster)}")
            if output == True:
                outputs.append(list(metabolites_in_cluster))
        
        
        cluster_col = pd.Series(self.col_labels, index=self.x.columns)

        group_means = pd.DataFrame(self.X_scaled, columns=self.x.columns, index=self.x.index)
        group_means['group'] = self.y

        results = []
        for c in cluster_col.unique():
            members = cluster_col[cluster_col == c].index
            for grp in sorted(set(self.y)):
                vals = group_means.loc[group_means['group'] == grp, members].values.flatten()
                results.append({'cluster': c, 'group': grp, 'mean_z': vals.mean()})

        cluster_by_group = pd.DataFrame(results).pivot(index='cluster', columns='group', values='mean_z')
        
        if output == True:
            return outputs, cluster_by_group
        else:
            return cluster_by_group
        
    def show_purity_table(self):
        """
        Shows how the mapped clusters compare to the true groups.

        Returns
        -------
        Must have run init and perform_cluster first.
        """
        purity = pd.crosstab(self.y, self.row_labels)
        print(purity)

    def show_heatmap(self):
        """
        Creates a heatmap showing the clustered datapoints and metabolites, coloured by their value. Displays
        the true group on a coloured bar, and shows a dendogram of the generated clusters. This clustermap
        is generated by seaborn, and is not the same clustering as done by the algorithm, although it should be
        similar as they both cluster the same thing. This is a good visualisation of what has happened, but it
        does not actually show the clusters that have been found. See "plot_biclusters" for the found clusters.

        Returns
        -------
        Must have run init and perform_cluster first.
        """
        group_labels = sorted(set(self.y))
        palette = sb.color_palette("tab10", len(group_labels))
        color_map = dict(zip(group_labels, palette))
        row_colors = pd.Series(self.y, index=self.x.index).map(color_map).to_numpy()

        g = sb.clustermap(self.X_scaled, row_colors=row_colors, cmap = "rocket", figsize = (20, 5))
        
        legend_handles = [mpatches.Patch(color=color_map[label], label=str(label)) for label in group_labels]

        g.ax_heatmap.legend(
            handles=legend_handles,
            title="Group",
            bbox_to_anchor=(1.10, 1),   # push right of the heatmap
            loc='upper left',
            borderaxespad=0.
            )
        g.fig.subplots_adjust(right=0.75)   # shrink the heatmap to make room on the right
        plt.show()
        
        self.g = g
        
    def heatmap_groupings(self, n_clusters_visual = 9, show = True, output = False):
        """
        Cuts the heatmap dendogram to create the number of cluster  given. Then prints the metabolites in
        each cluster. These are the clusters from the seaborn heatmap, not the clusters from the original
        spectral biclustering.

        Parameters
        ----------
        n_clusters_visual : int, optional
            The number of clusters you see in the column dendogram of the heatmap. The default is 9.
        show : bool, optional
            If True, prints values. Default is True
        output : bool, optional
            If True, returns information as a list. Default is False
        
        Returns
        -------
        Requires show_heatmap
        """
        

        outputs = []
        
        # g is your existing clustermap object
        col_linkage = self.g.dendrogram_col.linkage

        # Cut the dendrogram into n_clusters_visual clusters (pick a number that matches what you see visually)
        n_clusters_visual = n_clusters_visual
        visual_col_clusters = fcluster(col_linkage, t=n_clusters_visual, criterion='maxclust')

        # Map back to metabolite names — clustermap reorders columns,
        # so use g.data2d.columns (post-reordering) not the original x.columns
        metabolite_names_ordered = self.g.data2d.columns
        cluster_membership = pd.Series(visual_col_clusters, index=metabolite_names_ordered)

        for c in sorted(cluster_membership.unique()):
            members = cluster_membership[cluster_membership == c].index.tolist()
            if show == True:
                print(f"Cluster {c} (n={len(members)}): {members[:10]}...")
            if output == True:
                outputs.append(members)
        if output == True:
            return outputs
        
        
    def compare_silhouettes(self):
        """
        Generates silhouette scores for different numbers of row clusters and column clusters. When comparing
        row clusters, uses the number of column clusters used in the perform_cluster program and visa versa.
        Allows motivation of clustering parameter choices.

        Returns
        -------
        Requires init and perform_cluster
        """
        
    
        fig, axs = plt.subplots(2, 1, figsize = (10, 16))
        
        col_scores = []
        col_range = range(2, 20)
        for k in col_range:
            m = SpectralBiclustering(n_clusters=(self.n_row_clusters, k), method='log', random_state=self.random_state)
            m.fit(self.X_scaled)
            # silhouette needs the data transposed so columns become "samples" for this metric
            score = silhouette_score(self.X_scaled.T, m.column_labels_)
            col_scores.append(score)

        axs[0].set_title(f"Silhouette Figure for no. of rows = {self.n_row_clusters}")
        axs[0].plot(col_range, col_scores, marker='o')
        axs[0].set_xlabel("n_col_clusters")
        axs[0].set_ylabel("Silhouette score")

        row_scores = []
        row_range = range(2, 20)
        for a in row_range:
            n = SpectralBiclustering(n_clusters=(a, self.n_col_clusters), method='log', random_state=self.random_state)
            n.fit(self.X_scaled)
            # silhouette needs the data transposed so columns become "samples" for this metric
            score = silhouette_score(self.X_scaled, n.row_labels_)
            row_scores.append(score)

        axs[1].set_title(f"Silhouette Figure for no. of cols = {self.n_col_clusters}")
        axs[1].plot(row_range, row_scores, marker='o')
        axs[1].set_xlabel("n_row_clusters")
        axs[1].set_ylabel("Silhouette score")
        plt.show()

    # Looking at the 2 groups that have been best clustered by the algorithm
    def stats_tests(self, group_a = "Diabetic_Female", group_b = "Diabetic_Male"):
        """
        Compares how 2 true groups have been clustered. Uses Mann-Whitney U test to compare groups,
        generating a p value and a fold change between the 2. Also finds significant q values.

        Parameters
        ----------
        group_a : string/numeric, optional
            The name of a true group. The default is "AdSKO-hB2".
        group_b : string/numeric, optional
            The name of a true group. The default is "SKO-PBS".

        Returns
        -------
        Requires init and perform_cluster
        """
        
    
        # Isolate the two groups of interest (use your actual known-label values)
        group_a_mask = (self.y == group_a)
        group_b_mask = (self.y == group_b)
        
        X_a = self.x_numeric[group_a_mask]
        X_b = self.x_numeric[group_b_mask]

        # Test whether group a is consistently higher or lower than group b

        results = []
        for col in self.x_numeric.columns:
            stat, pval = stats.mannwhitneyu(X_a[col], X_b[col], alternative='two-sided')  # stat = the U statistic, pval is the pval from two_sided
            fold_change = X_a[col].mean() / X_b[col].mean()  # Mean intensity between the 2
            results.append({'metabolite': col, 'p_value': pval, 'fold_change': fold_change})

        results_df = pd.DataFrame(results).sort_values('p_value')

        

        # We expect about 0.05 to be false positives anyway, so look for where the value is lower

        results_df['q_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        significant = results_df[results_df['q_value'] < 0.05]

        print(f"The significant results were: {significant}")

        purple_rows = np.where(group_b_mask)[0]
        mean_by_colcluster = pd.DataFrame({
            'col_cluster': self.col_labels,
            'mean_intensity_purple': self.X_scaled[purple_rows].mean(axis=0),
            'mean_intensity_rest': self.X_scaled[~group_b_mask].mean(axis=0)
            }).groupby('col_cluster').mean()
        print(mean_by_colcluster)
        
    def row_col_cluster_matrix(self):
        """
        Builds a (n_row_clusters x n_col_clusters) matrix of mean z-scores,
        i.e. how much each metabolite module is elevated/reduced in each
        sample (row) cluster. This is the direct bicluster-level link.
        """
        mat = np.zeros((self.n_row_clusters, self.n_col_clusters))
        for r in range(self.n_row_clusters):
            for c in range(self.n_col_clusters):
                block = self.X_scaled[self.row_labels == r][:, self.col_labels == c]
                mat[r, c] = block.mean() if block.size else np.nan

        return pd.DataFrame(
            mat,
            index=[f"row_cluster_{r}" for r in range(self.n_row_clusters)],
            columns=[f"col_cluster_{c}" for c in range(self.n_col_clusters)]
            )
    
    def compare_msr(self, row_range=range(2, 15), col_range=range(2, 20)):
        """
        Sweeps n_col_clusters (holding n_row_clusters fixed) and n_row_clusters
        (holding n_col_clusters fixed), plotting overall MSR (Mean Squared Residue) at each step.
        Lower MSR = more coherent biclusters, so look for an elbow/plateau
        rather than the global minimum (MSR trivially -> 0 as clusters shrink
                                        towards single cells).
        More useful for biclustering than silhouette score, as built for this.
        
        Parameters
        ----------
        row_range : range object, optional
            The row_range to be swept over. The default is range(2, 15)
        col_range : range object, optional
            The col_range to be swept over. The default is range(2, 20)
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        col_scores = []
        for k in col_range:
            m = SpectralBiclustering(n_clusters=(self.n_row_clusters, k),
                                     method='log', random_state=self.random_state)
            m.fit(self.X_scaled)
            score = self._msr_for_labels(m.row_labels_, m.column_labels_,
                                         self.n_row_clusters, k)
            col_scores.append(score)

        axs[0].plot(list(col_range), col_scores, marker='o')
        axs[0].set_title(f"MSR vs n_col_clusters (n_row_clusters={self.n_row_clusters})")
        axs[0].set_xlabel("n_col_clusters"); axs[0].set_ylabel("Mean Squared Residue")

        row_scores = []
        for k in row_range:
            m = SpectralBiclustering(n_clusters=(k, self.n_col_clusters),
                                     method='log', random_state=self.random_state)
            m.fit(self.X_scaled)
            score = self._msr_for_labels(m.row_labels_, m.column_labels_,
                                      k, self.n_col_clusters)
            row_scores.append(score)

        axs[1].plot(list(row_range), row_scores, marker='o')
        axs[1].set_title(f"MSR vs n_row_clusters (n_col_clusters={self.n_col_clusters})")
        axs[1].set_xlabel("n_row_clusters"); axs[1].set_ylabel("Mean Squared Residue")

        plt.tight_layout()
        plt.show()

    def mean_squared_residue(self, per_bicluster=False):
        """
        Computes the Mean Squared Residue (Cheng Y, Church GM. Biclustering of 
        expression data. Proc Int Conf Intell Syst Mol Biol. 2000;8:93-103. PMID: 10977070.) for each
        (row_cluster, col_cluster) block from the current biclustering, and
        the overall weighted average MSR across all blocks. Lower = more
        internally coherent checkerboard structure.

        Parameters
        ----------
        per_bicluster : bool, optional
            If True, also returns a DataFrame of MSR per block. Default False.

        Returns
        -------
        overall_msr : float
            Size-weighted average MSR across all blocks.
        block_msr : pd.DataFrame (only if per_bicluster=True)
            MSR, n_rows, n_cols per (row_cluster, col_cluster) block.
        """
        X = self.X_scaled
        records = []
        total_sq_resid = 0.0
        total_n = 0

        for r in range(self.n_row_clusters):
            row_mask = self.row_labels == r
            for c in range(self.n_col_clusters):
                col_mask = self.col_labels == c
                block = X[np.ix_(row_mask, col_mask)]

                if block.size == 0:
                    continue

                row_means = block.mean(axis=1, keepdims=True)   # x_iJ, per row
                col_means = block.mean(axis=0, keepdims=True)   # x_Ij, per col
                overall_mean = block.mean()                      # x_IJ

                residues = block - row_means - col_means + overall_mean
                sq_resid = residues ** 2

                msr = sq_resid.mean()
                n_entries = block.size

                records.append({
                    'row_cluster': r, 'col_cluster': c,
                    'n_rows': row_mask.sum(), 'n_cols': col_mask.sum(),
                    'msr': msr
                    })

                total_sq_resid += sq_resid.sum()
                total_n += n_entries

        overall_msr = total_sq_resid / total_n
        block_msr = pd.DataFrame(records)

        if per_bicluster:
            return overall_msr, block_msr
        return overall_msr
    def _msr_for_labels(self, row_labels, col_labels, n_row, n_col):
        """Helper: MSR for an arbitrary label assignment, not just self.row_labels/col_labels."""
        X = self.X_scaled
        total_sq_resid, total_n = 0.0, 0
        for r in range(n_row):
            row_mask = row_labels == r
            for c in range(n_col):
                col_mask = col_labels == c
                block = X[np.ix_(row_mask, col_mask)]
                if block.size == 0:
                    continue
                row_means = block.mean(axis=1, keepdims=True)
                col_means = block.mean(axis=0, keepdims=True)
                overall_mean = block.mean()
                residues = block - row_means - col_means + overall_mean
                total_sq_resid += (residues ** 2).sum()
                total_n += block.size
        return total_sq_resid / total_n
    
    def run_everything(self, params = None):
        """
        A simple function to run the whole code. It is recommended to create your own pipeline with the
        methods you want to use, letting you set all of the parameters.

        Parameters
        ----------
        params : list, optional
            A list of lists, containing the parameters to be used. If used, there must be
            6 contained lists. 
            params[0] = [n_row_clusters, n_col_clusters] for use in perform_cluster
            params[1] = [cmap, vmin, vmax, figsize] for use in plot_biclusters
            params[2] = [show, output] for use in cluster_groupings
            params[3] = [n_clusters_visual, show, output] for use in heatmap_groupings
            params[4] = [group_a, group_b] for use in stats_tests
            params[5] = [row_range, col_range] for use in compare_msr
            For the requirements for the parameters in the lists, refer to the method text.
            The default is None.

        Returns
        -------
        None
        """
        print("Perform the Spectral Biclustering algorithm, clustering the datapoints and the features.")
        next_step = input("\nWould you like to run? y/n")
        if next_step == "y":
            try: 
                n_row_clusters, n_col_clusters = params[0]
                self.perform_cluster(n_row_clusters, n_col_clusters)
            except:
                self.perform_cluster()
        print("\nPlots the results of the biclustering algorithm as a heatmap.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":        
            try: 
                cmap, vmin, vmax, figsize = params[1]
                self.plot_biclusters(cmap, vmin, vmax, figsize)
            except:
                self.plot_biclusters()
        print("\nReturns the clusters of the metabolite groups.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try: 
                show, output = params[2]
                self.cluster_groupings(show, output)
            except:
                self.cluster_groupings()
        print("\nShows the overlap between clustered datapoints and true groups.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
                self.show_purity_table()
        print("\nGenerate a seaborn clustermap to approximately visualise the clustering algorithm.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
                self.show_heatmap()
        print("\nFinds the feature groupings from the seaborn heatmap by cutting the dendogram.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try:
                n_clusters_visual, show, output = params[3]
                self.heatmap_groupings(n_clusters_visual, show, output)
            except:
                self.heatmap_groupings()
        print("\nCompares the silhouette scores for varied numbers of row and column clusters.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
                self.compare_silhouettes()
        print("\nPerforms statistical tests on the clustering to see the effectiveness of the model.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try:
                group_a, group_b = params[4]
                self.stats_tests(group_a, group_b)
            except:
                self.stats_tests()
        print("\nA table of the z values between the row and column clusters.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
                self.row_col_cluster_matrix()
        print("\nCompares the mean squared residual for varied numbers of row and column clusters.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try:
                row_range, col_range = params[5]
                self.compare_msr(row_range, col_range)
            except:
                self.compare_msr()

if __name__ == "__main__":
    SB = Spectral_Biclustering(data_func = load_human_metabolites, data_frame = True, )
    SB.run_everything()
