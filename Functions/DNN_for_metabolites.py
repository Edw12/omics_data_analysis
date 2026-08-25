# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:17:13 2026

@author: Edwin
"""
import keras
from keras import layers
from keras import regularizers

from sklearn.feature_selection import mutual_info_regression

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score, roc_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import seaborn as sns
from xgboost import XGBClassifier
import tqdm
import shap

class DNN_for_Metabolites:
    def __init__(self, seed = 42):
        self.seed = seed
        
    def generate_data(self, num_datapoints = 100000, num_metabolites = 300, num_groups = 8):
        """
        Generates the sample data. It will be a pandas df of num_datapoints x num_metabolites

        Parameters
        ----------
        num_datapoints : int, optional
            The number of datapoints to be generated. The default is 100000.
        num_metabolites : int, optional
            The number of features for the data. The default is 300.
        num_groups : int, optional
            The number of groups for the data. The default is 8.
        """

        np.random.seed(self.seed)  # remove for different data each run

        # ----------------------------
        # Parameters
        # ----------------------------
        num_datapoints = 100000     # 30000 gives a good view of things, need ~50000 to get good discovery of groups 1, 2, 6, 7, beyond that can be overkill, although get better at finding group 3, 4, 5
        num_metabolites = 300
        num_groups = 8
        n_per_group = num_datapoints // num_groups  # 250 per group
        
        self.num_groups = num_groups
        
        
        datapoint_names = [f'Sample_{i+1:04d}' for i in range(num_datapoints)]
        metabolite_names = [f'Metabolite_{j+1:03d}' for j in range(num_metabolites)]

        # ----------------------------
        # 1. Balanced group labels, shuffled
        # ----------------------------
        group_names = [f'Group_{k+1}' for k in range(num_groups)]
        labels = np.repeat(group_names, n_per_group)
        np.random.shuffle(labels)
        y_data = pd.DataFrame(labels, index=datapoint_names, columns=['Group'])
      
        def group_mask(name):
          return (y_data['Group'] == name).values

        # ----------------------------
        # 2. Start with pure background noise everywhere
        #    (most columns will carry zero signal - that's the point)
        # ----------------------------
        x_data_values = np.random.rand(num_datapoints, num_metabolites) * 100
        x_data = pd.DataFrame(x_data_values, index=datapoint_names, columns=metabolite_names)
        
        # --- Group 1: single variable, monotonic threshold ---
        # high Metabolite_001 -> Group_1
        mask = group_mask('Group_1')
        x_data.loc[mask, 'Metabolite_001'] = np.random.normal(loc = 85, scale = 8, size = mask.sum()).clip(0, 100)
        x_data.loc[~mask, 'Metabolite_001'] = np.random.normal(loc = 45, scale = 20, size = (~mask).sum()).clip(0, 100)
      
        # --- Group 2: linear dependence between 2 variables ---
        mask = group_mask('Group_2')
        n = mask.sum()
        base = np.random.normal(75, 10, n)
        x_data.loc[mask, 'Metabolite_002'] = (base + np.random.normal(0, 8, n)).clip(0, 100)
        x_data.loc[mask, 'Metabolite_003'] = (0.7*base + 0.3*np.random.normal(75, 10, n)).clip(0, 100)
      
        # --- Group 3: non-linear (radial) relationship between 2 variables ---
        mask = group_mask('Group_3')
        n = mask.sum()
        theta = np.random.uniform(0, 2*np.pi, n)
        radius = np.random.normal(30, 4, n)
        x_data.loc[mask, 'Metabolite_004'] = (50 + radius*np.cos(theta)).clip(0, 100)
        x_data.loc[mask, 'Metabolite_005'] = (50 + radius*np.sin(theta)).clip(0, 100)
      
        # --- Group 4: XOR-style interaction (no single variable predicts it) ---
        mask = group_mask('Group_4')
        n = mask.sum()
        side_6 = np.random.rand(n) < 0.5
        side_7 = np.random.rand(n) < 0.5   # independent flip
        m6 = np.where(side_6, np.random.normal(75, 10, n), np.random.normal(25, 10, n))
        m7 = np.where(side_7, np.random.normal(75, 10, n), np.random.normal(25, 10, n))
        x_data.loc[mask, 'Metabolite_006'] = np.clip(m6, 0, 100)
        x_data.loc[mask, 'Metabolite_007'] = np.clip(m7, 0, 100)
        
        # --- Group 5: periodic/sinusoidal relationship between 2 variables ---
        mask = group_mask('Group_5')
        n = mask.sum()
        m8 = np.random.uniform(0, 100, n)
        m9 = 50 + 35*np.sin(m8/100*4*np.pi) + np.random.normal(0, 5, n)
        x_data.loc[mask, 'Metabolite_008'] = m8
        x_data.loc[mask, 'Metabolite_009'] = np.clip(m9, 0, 100)
      
        # --- Group 6: distributed linear signal across 5 variables (weak individually) ---
        mask = group_mask('Group_6')
        n = mask.sum()
        weights = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
        cols = [f'Metabolite_{i:03d}' for i in range(10, 15)]
        target_sum = np.random.normal(70, 6, n)
        for w, c in zip(weights, cols):
            x_data.loc[mask, c] = np.clip(target_sum*w/weights.sum() + np.random.normal(0, 10, n), 0, 100)

        # --- Group 7: high-dimensional non-linear cluster (5-D centroid distance) ---
        mask = group_mask('Group_7')
        n = mask.sum()
        cols = [f'Metabolite_{i:03d}' for i in range(15, 20)]
        centroid = np.array([40, 60, 40, 60, 50])
        pts = centroid + np.random.normal(0, 6, size=(n, 5))
        for i, c in enumerate(cols):
            x_data.loc[mask, c] = np.clip(pts[:, i], 0, 100)
          
        # --- Group 8: pure noise / negative control ---
        # left untouched - tests that the network doesn't hallucinate signal where none exists
          
        print(y_data['Group'].value_counts())
        self.x_data, self.y_data = x_data, y_data
  
    def encode_labels(self):
        # Encode categorical labels to integers
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y_data)
        self.x_train, x_test, self.y_train, y_test = train_test_split(self.x_data, self.y_encoded, test_size = 0.3, random_state = self.seed)
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = self.seed)
    
    def plot_seeded_values(self, n_samples = 5000):
        """
        Plots the seeded data to allow observation. Often too tightly packed to see for no. of datapoints
        necissary for good training so only samples a subset.
        
        Parameters
        ----------
        n_samples : int, optional
            The number of points per group that will be plotted on each axis. Each axis will have 
            n_samples*n_groups plotted.

        Returns
        -------
        None
        """
        def observe_axis(x_axis, y_axis, ax, alpha = 1.0):
            # Get unique integer encoded labels and their corresponding original string labels
            unique_encoded_groups = np.unique(self.y_encoded)
            original_group_names = self.label_encoder.inverse_transform(unique_encoded_groups)

            for i, encoded_group in enumerate(unique_encoded_groups):
                # Filter X_pca using the integer encoded labels for proper indexing
                ax.scatter(self.x_data.loc[self.y_encoded == encoded_group, x_axis].sample(n = n_samples, random_state = self.seed),
                           self.x_data.loc[self.y_encoded == encoded_group, y_axis].sample(n = n_samples, random_state = self.seed),
                           label = original_group_names[i],   # Use original string label for legend
                           alpha = alpha)
            ax.legend() # Changed to ax.legend()
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'{x_axis} vs {y_axis} by Group')

        def observe_pca_axis(pca_df, ax, group_num, alpha = 1.0):
            # Get the globally defined ordered group names for consistent legend order
            unique_encoded_groups = np.unique(self.y_encoded)
            original_group_names = self.label_encoder.inverse_transform(unique_encoded_groups)

            for group_name in original_group_names: # Iterate in the desired order
                mask = pca_df["Group"] == group_name
                # Only plot if the group exists in the current pca_df (it should if all groups are concatenated)
                if mask.any():
                    ax.scatter(pca_df.loc[mask, "PC1"].samples(n = n_samples, random_state = self.seed),
                               pca_df.loc[mask, "PC2"].samples(n = n_samples, random_state = self.seed),
                               label=group_name,
                               alpha = alpha)
            ax.legend()
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Group {group_num} PCA")
        
        num_plots = self.num_groups

        pca = PCA(n_components = 2, random_state = self.seed)
        alpha = 0.2

        # As groups 6 and 7 are based on multiple variables, in order to see the pattern of their seed, they are plotted in the principal components of those variables

        cols = [f'Metabolite_{i:03d}' for i in range(10, 15)]
        x_pca_6_transformed = pca.fit_transform(self.x_data[cols])
        X_pca_6 = pd.DataFrame(x_pca_6_transformed,columns=["PC1","PC2"], index = self.y_data.index)
        X_pca_6 = pd.concat([X_pca_6, self.y_data], axis=1, join="inner")

        cols = [f'Metabolite_{i:03d}' for i in range(15, 20)]
        x_pca_7_transformed = pca.fit_transform(self.x_data[cols])
        X_pca_7 = pd.DataFrame(x_pca_7_transformed,columns=["PC1","PC2"], index = self.y_data.index)
        X_pca_7 = pd.concat([X_pca_7, self.y_data], axis=1, join="inner")

        seeds = {"1" : ["Metabolite_001", "Metabolite_002"],
                 "2" : ["Metabolite_002", "Metabolite_003"],
                 "3" : ["Metabolite_004", "Metabolite_005"],
                 "4" : ["Metabolite_006", "Metabolite_007"],
                 "5" : ["Metabolite_008", "Metabolite_009"],
                 }

        fig, axs = plt.subplots(int(num_plots/2), 2, figsize = (16, 16))
        fig.suptitle("Showing Seeded relationships")

        for group_num in range(1, self.num_groups + 1):
            ax = axs[(group_num-1)//2, (group_num-1)%2]

            if group_num in [1, 2, 3, 4, 5]:
                observe_axis(seeds[str(group_num)][0], seeds[str(group_num)][1], ax, alpha = alpha)  # Only depend on 2 variables
            elif group_num == 6:
                observe_pca_axis(X_pca_6, ax, group_num, alpha = alpha)
            elif group_num == 7:
                observe_pca_axis(X_pca_7, ax, group_num)
            elif group_num == 8:
                # For Group 8 (pure noise), plot a generic scatter to fill the subplot
                # For example, using Metabolite_001 and Metabolite_002 for this group only
                mask_group_8 = (self.y_data['Group'] == f'Group_{group_num}').values
                ax.scatter(self.x_data.loc[mask_group_8, 'Metabolite_001'],
                                self.x_data.loc[mask_group_8, 'Metabolite_002'],
                                label = f'Group_{group_num}')
                ax.set_title(f'Group {group_num}: Pure Noise (M_001 vs M_002)')
                ax.set_xlabel('Metabolite_001')
                ax.set_ylabel('Metabolite_002')
                ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()
        
        
    def preclustering_for_MI(self):
        """
        Clusters using an XGBClassifier to find important features from the start.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        clf = XGBClassifier(n_estimators=150,
                    tree_method = "hist", n_jobs = -1,
                    max_depth=6, random_state=self.seed)
        clf.fit(self.x_train, self.y_train)
        
        top_feature_indices = np.argsort(clf.feature_importances_)[::-1][:20]
        top_cols = self.x_data.columns[top_feature_indices]
        
        def pairwise_mi_matrix(df, cols):
            n = len(cols)
            mi_matrix = np.zeros((n, n))
            for i, c1 in enumerate(cols):
                # mutual info of column c1 against all other columns at once
                mi_matrix[i, :] = mutual_info_regression(df[cols], df[c1], random_state=self.seed)
            return pd.DataFrame(mi_matrix, index=cols, columns=cols)

        mi_df = pairwise_mi_matrix(self.x_data, top_cols)

        mask = np.eye(len(mi_df), dtype=bool)

        plt.figure(figsize=(12, 10))
        sns.heatmap(mi_df, cmap='viridis', mask=mask, annot=False)
        plt.title('Pairwise mutual information (diagonal excluded)')
        plt.show()
        
        self.mi_df = mi_df
        
        return mi_df
    
    def Mutual_Information_sweep(self):
        """
        Creates a Mutual_Information plot for use in creating important features.

        Returns
        -------
        None
        """
        # normalize MI to [0, 1] using each column's self-MI (max possible) as scale
        diag = np.diag(self.mi_df.values).copy() # Make a writeable copy
        diag[diag == 0] = 1e-10  # avoid divide-by-zero for columns with no signal at all
        norm = self.mi_df.values / np.sqrt(np.outer(diag, diag))
        norm = np.clip(norm, 0, 1)
        dist = 1 - norm
        np.fill_diagonal(dist, 0)
        # force symmetry (MI matrix from mutual_info_regression isn't perfectly symmetric numerically)
        dist = (dist + dist.T) / 2
        self.dist_df = pd.DataFrame(dist, index=self.mi_df.index, columns=self.mi_df.columns)

        # condensed distance matrix required by scipy's linkage
        condensed = squareform(self.dist_df.values, checks=False)
        self.Z = linkage(condensed, method='average') # Store Z as an instance attribute

        plt.figure(figsize=(14, 6))
        dendrogram(self.Z, labels=self.dist_df.columns.tolist(), leaf_rotation=90)
        plt.title('Metabolite clusters by mutual information')
        plt.ylabel('Distance (1 - normalized MI)')
        plt.tight_layout()
        plt.show()

    def suggest_cluster_threshold(self):
        """
        Suggests the threshold to cut the dendogram, recommended to check the dendogram first.
        """
        merge_heights = np.sort(self.Z[:, 2])
        gaps = np.diff(merge_heights)
        biggest_gap_idx = np.argmax(gaps)
        return merge_heights[biggest_gap_idx] + gaps[biggest_gap_idx]/2

    def cluster(self, threshold = 0.7):
        """
        Finds the clusters shown on the dendogram, and stores them in self.multi_clusters.
        Threshold gives how high the branch needs to be to be allowed, 0 accepts nothing, 1 groups all into 1 cluster
        """
        # distance threshold: tune based on where the dendrogram shows a natural gap
        
        cluster_labels = fcluster(self.Z, t=threshold, criterion='distance')

        cluster_df = pd.DataFrame({'metabolite': self.dist_df.columns, 'cluster': cluster_labels})

        # only show clusters with more than 1 member (single-column clusters = no partner found)
        multi_clusters = cluster_df.groupby('cluster').filter(lambda g: len(g) > 1)
        for cid, group in multi_clusters.groupby('cluster'):
            print(f"Cluster {cid}: {list(group['metabolite'])}")

        self.multi_clusters = multi_clusters

    def add_features(self):
        """
        Using the found clusters, generates features to augment the data to make the relationship easier for the DNN to find
        """
        new_features = {}
        if hasattr(self, 'multi_clusters') and not self.multi_clusters.empty:
            for cid, group in self.multi_clusters.groupby('cluster'):
                cols = list(group['metabolite'])
                if len(cols) == 2:
                    c1, c2 = cols
                    a, b = self.x_data[c1], self.x_data[c2]
                    new_features[f'cluster{cid}_prod'] = (a - a.mean()) * (b - b.mean())
                    new_features[f'cluster{cid}_dist']  = np.sqrt((a - a.mean())**2 + (b - b.mean())**2)
                else:
                    # for clusters with 3+ columns, distance from the cluster's own centroid
                    sub = self.x_data[cols]
                    centroid = sub.mean()
                    new_features[f'cluster{cid}_centroid_dist'] = np.sqrt(((sub - centroid) ** 2).sum(axis=1))
        
        self.new_features_df = pd.DataFrame(new_features, index=self.x_data.index)
        
        self.x_data_aug = pd.concat([self.x_data, self.new_features_df], axis=1)

        return self.x_data_aug, self.new_features_df
    
    def finish_preprocessing(self, add_features = True):
        """
        Scales the data and creates df versions of the data. If add_features = True, adds the features created
        using the pre-training clustering
        
        Returns
        -------
        None
        """
        if add_features == True:
            x_train = self.x_train.join(self.new_features_df)
            x_test = self.x_test.join(self.new_features_df)
            x_val = self.x_val.join(self.new_features_df)
            columns = np.concat([self.x_data.columns, self.new_features_df.columns])
        else:
            x_train = self.x_train
            x_test = self.x_test
            x_val = self.x_val
            columns = self.x_data.columns
        
        
        standard_scaler = StandardScaler()
        self.x_train = standard_scaler.fit_transform(x_train)  # Work out the different scaling using the train data
        self.x_test = standard_scaler.transform(x_test)        # Apply same scaling to the test data
        self.x_val = standard_scaler.transform(x_val)          # Apply same scaling to the validation data
        
        self.x_train_df = pd.DataFrame(x_train, columns = columns)
        self.x_test_df = pd.DataFrame(x_test, columns = columns)
        self.x_val_df = pd.DataFrame(x_val, columns = columns)
        
    def RF_on_created_columns(self):
        """
        Uses a quick Random Forest to show the effectiveness of the generated columns in identifying the
        groups without the added noise of the other classes.
        """
        new_names = self.new_features_df.columns.tolist()
        subset_train = self.x_train_df[new_names]
        subset_test = self.x_test_df[new_names]

        classifier = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        classifier.fit(subset_train, self.y_train)
        y_pred = classifier.predict(subset_test)

        cm = ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred)
        plt.title("Quick Random Forest")
        plt.show()
        
    def create_model(self, l1 = 1e-5, l2 = 1e-4):
        """
        Creates the model, with adjustable loss functions

        Parameters
        ----------
        l1 : float, optional
            Drives irrelevant feature coefficients to 0. The default is 1e-5.
        l2 : float, optional
            Shrinks all weights evenly. The default is 1e-4.

        Returns
        -------
        None
        """

        # L1 (Lasso) uses absolute weight values to drive irrelevant feature coefficients to zero,
        # creating sparse, interpretable models. L2 (Ridge) uses squared weight values to shrink all weights evenly without zeroing them out

        inputs = keras.Input(shape=(self.x_train.shape[1],), name = "meta-inputs")
        features_1 = layers.Dense(256, activation = "relu", kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2))(inputs)
        features_1 = layers.Dropout(0.1)(features_1)
        features_2 = layers.Dense(128, activation = "swish", kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2))(features_1)
        features_2 = layers.Dropout(0.1)(features_2)
        features_3 = layers.Dense(32, activation = "relu", kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2))(features_2)

        outputs = layers.Dense(8, activation = "softmax", name = "meta-outputs")(features_3)

        self.model = keras.Model(inputs = inputs, outputs = outputs, name = "meta-model")
        
        self.model.summary()
        
    def train_model(self):
        """
        Trains the model on the train data.

        Returns
        -------
        None
        """
        class PerClassAccuracy(keras.callbacks.Callback):
            def __init__(self, x_val, y_val, class_names):
                super().__init__()
                self.x_val = x_val
                self.y_val = y_val
                self.class_names = class_names
                self.history = {name: [] for name in class_names}

            def on_epoch_end(self, epoch, logs=None):
                preds = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
                cm = confusion_matrix(self.y_val, preds, labels=range(len(self.class_names)))
                per_class_acc = cm.diagonal() / cm.sum(axis=1)

                for name, acc in zip(self.class_names, per_class_acc):
                    self.history[name].append(acc)
                    
        

        class MacroF1EarlyStopping(keras.callbacks.Callback):
            def __init__(self, x_val, y_val, patience=20):
                super().__init__()
                self.x_val, self.y_val = x_val, y_val
                self.patience = patience
                self.best_f1 = -1
                self.best_epoch = -1
                self.saved_vals = []
                self.wait = 0
                self.best_weights = None

            def on_epoch_end(self, epoch, logs=None):
                preds = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
                f1 = f1_score(self.y_val, preds, average='macro')
                self.saved_vals.append(f1)
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_epoch = epoch
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                    self.model.save('best_model_by_f1.keras')   # <-- persist best-F1 checkpoint
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)
                print(f" — val_macro_f1: {f1:.4f} (best: {self.best_f1:.4f})")
        
        self.class_names = self.label_encoder.classes_
        self.per_class_cb = PerClassAccuracy(self.x_val, self.y_val, self.class_names)  # x_val/y_val = your held-out validation split

        self.F1_callback = MacroF1EarlyStopping(x_val=self.x_val, y_val=self.y_val, patience = 20)

        self.model.compile(
            optimizer = "adam",
            loss = "sparse_categorical_crossentropy", # Use single appropriate loss
            metrics = ["accuracy"], # Use single appropriate metric
            )
        self.model.fit(
                self.x_train,
                self.y_train,
                epochs = 100,
                verbose = True,
                validation_data = (self.x_val, self.y_val),
                callbacks = [self.per_class_cb, self.F1_callback],
                )
        self.model.evaluate(
                self.x_test,
                self.y_test
                )

        # Predict using the model (outputs will be probabilities, not separate preds for priority/department)
        self.predictions = self.model.predict(
                [self.x_test]
                )
        
    def model_accuracy(self):
        """
        Plots the validation metrics of the model over the epochs. Then plots the accuracy per group over the
        epochs.

        Returns
        -------
        None
        """
        accuracy = self.model.history.history["accuracy"]
        loss = self.model.history.history["loss"]
        validation = self.model.history.history["val_loss"]
        val_accuracy = self.model.history.history["val_accuracy"]
        F1_accuracy = self.F1_callback.saved_vals

        x = np.arange(len(accuracy))

        fig, axs = plt.subplots(5, 1, figsize = (16, 8), sharex = True)

        axs[0].plot(x, accuracy)
        axs[1].plot(x, loss)
        axs[2].plot(x, validation)
        axs[3].plot(x, val_accuracy)
        axs[4].plot(x, F1_accuracy)
        axs[4].axvline(self.F1_callback.best_epoch, color='black', linestyle='--', linewidth=1, alpha=0.7,
                       label=f'Best F1 (epoch {self.F1_callback.best_epoch})')

        axs[3].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[1].set_ylabel("Loss")
        axs[2].set_ylabel("Validation Loss")
        axs[3].set_ylabel("Validation Accuracy")
        axs[4].set_ylabel("F1 Score")

        axs[2].set_xticks(ticks = x, minor = True)

        fig.suptitle("Comparing success with Epoch")
            
        plt.figure(figsize=(10, 6))
        for name in self.class_names:
            plt.plot(self.per_class_cb.history[name], label=name)

        plt.axvline(self.F1_callback.best_epoch, color='black', linestyle='--', linewidth=1, alpha=0.7,
                    label=f'Best F1 (epoch {self.F1_callback.best_epoch})')

        plt.xlabel('Epoch')
        plt.ylabel('Per-class accuracy')
        plt.legend()
        plt.title('Per-group accuracy over training')
        plt.show()
            
    def observing_results(self):
        """
        Gives information about the correct allocation of datapoints to groups..

        Returns
        -------
        None
        """
        predicted_class_indices = np.argmax(self.predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_class_indices)
        true_labels = self.label_encoder.inverse_transform(self.y_test)

        correct = predicted_labels == true_labels

        seeds = {
            "Group_1": "1",
            "Group_2" : "2 Linear",
            "Group_3" : "2 Non-Linear",
            "Group_4" : "2 XOR",
            "Group_5" : "2 Sinusoidal",
            "Group_6" : "5 Linear",
            "Group_7" : "5 Non-Linear",
            "Group_8" : "Pure Noise"
            }

        print(f"{np.sum(correct)} out of {len(true_labels)} were successfully classified")
        print("Group: Classified as : in the dataset : correctly classified : Variables and relationship")
        for a in np.unique(true_labels):
            print(f"{a}: {np.sum(predicted_labels == a)} : {np.sum(true_labels == a)} : {np.sum(np.logical_and(predicted_labels == a, true_labels == a))} : {seeds[a]}")
            
        ConfusionMatrixDisplay.from_predictions(y_true = true_labels, y_pred = predicted_labels, cmap = "plasma")
        
    def ROC_curves(self, n_classes = 8):
       """
        Creates ROC curves per groups, average ROC curves, and shows a pure chance line.

        Returns
        -------
        None
        """

       # ----------------------------------------------------------------
       # 1. Get predicted probabilities and true labels
       # ----------------------------------------------------------------
       y_pred_proba = self.model.predict(self.x_test, verbose=0)  # shape: (n_samples, 8)

       class_names = [f'Group_{k+1}' for k in range(n_classes)]

       # Binarize the true labels for one-vs-rest ROC (y_test must be integer-encoded 0-7)
       y_test_bin = label_binarize(self.y_test, classes=range(n_classes))

       # ----------------------------------------------------------------
       # 2. Compute ROC curve and AUC for each class
       # ----------------------------------------------------------------
       fpr = {}
       tpr = {}
       roc_auc = {}

       for i in range(n_classes):
           fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
           roc_auc[i] = auc(fpr[i], tpr[i])

       # ----------------------------------------------------------------
       # 3. Micro-average (aggregate across all classes/samples)
       # ----------------------------------------------------------------
       fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
       roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

       # ----------------------------------------------------------------
       # 4. Macro-average (unweighted mean across classes - fairer when
       #    classes are imbalanced or vary in difficulty, as here)
       # ----------------------------------------------------------------
       all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
       mean_tpr = np.zeros_like(all_fpr)
       for i in range(n_classes):
           mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
       mean_tpr /= n_classes

       fpr["macro"] = all_fpr
       tpr["macro"] = mean_tpr
       roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

       # ----------------------------------------------------------------
       # 5. Plot
       # ----------------------------------------------------------------
       plt.figure(figsize=(9, 8))

       # Per-class curves
       colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
       for i, color in zip(range(n_classes), colors):
           plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

       # Micro/macro averages, emphasized
       plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
       plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
                label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')

       # Chance line
       plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')

       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('ROC Curves — One-vs-Rest by Group')
       plt.legend(loc='lower right', fontsize=8)
       plt.tight_layout()
       plt.show() 
      
    def calc_SHAP(self, num_samples_to_explain = 100):
        """
        "SHAP (SHapley Additive exPlanations) 
        is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using Shapley values from cooperative game theory.
        Here's a brief overview of how SHAP works and what the visualizations represent:
        Shapley Values: For each prediction, SHAP calculates a value for every feature that represents how much
        that feature contributed to the prediction being different from the average prediction. A positive SHAP
        value means the feature increased the prediction, while a negative value decreased it.
        Summary Plot: This plot provides an overview of the most important features and their impact across all
        samples. Each dot represents a Shapley value for a feature and an instance. The color usually indicates
        the feature's value (e.g., red for high, blue for low).
        Dependence Plot: This plot shows how a single feature impacts the model's output. It plots the 
        feature's value against its SHAP value, often revealing non-linear relationships or interactions with 
        other features (which can be indicated by color).".  <--- SOURCE!

        Returns
        -------
        None
        """
        self.feature_names = self.x_data_aug.columns.tolist()
        # Ensure eager execution is enabled for SHAP
        #tf.compat.v1.enable_eager_execution()

        # Using GradientExplainer

        explainer = shap.GradientExplainer(self.model, self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)])

        # Calculate SHAP values for a subset of the test data for demonstration
        # This can take a while depending on the size of the subset and model complexity
        self.num_samples_to_explain = num_samples_to_explain
        self.samples_to_explain = self.x_test[np.random.choice(self.x_test.shape[0], self.num_samples_to_explain, replace=False)]
        self.shap_values = explainer.shap_values(self.samples_to_explain)

        print(f"SHAP values calculated for {self.num_samples_to_explain} test samples.")
        
        global_shap_values = np.mean(np.abs(np.array(self.shap_values)), axis=0)
        overall_importance_df = pd.DataFrame({
                                "feature": self.feature_names,
                                "mean_abs_shap_value": np.mean(global_shap_values, axis=1) # Changed axis from 0 to 1
                                }).sort_values("mean_abs_shap_value", ascending=False)
        print("\nOverall Feature Importance (Mean Absolute SHAP Value across all classes):\n")
        
        self.top_features = overall_importance_df['feature'].head(8).tolist()  # Find the top features by name
        
    def SHAP_plots(self):
        n_plots = len(self.class_names)

        for i in range(n_plots):
            plt.figure()
            shap.summary_plot(self.shap_values[:, :, i], self.samples_to_explain, feature_names=self.feature_names, show=False, cmap = "plasma")
            plt.title(f"SHAP Summary Plot for {self.class_names[i]}")

        plt.show()
        
    def SHAP_important_features(self):
        top_k = 8
        self.top_features_per_class = {}

        for i, cname in enumerate(self.class_names):
            mean_abs = np.abs(self.shap_values[:, :, i]).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:top_k]
            self.top_features_per_class[cname] = [self.feature_names[j] for j in order]
            print(cname, self.top_features_per_class[cname])
            
    def top_metabolite_pairplots(self):
        test_idx = np.random.choice(self.x_test.shape[0], self.num_samples_to_explain, replace=False)
        class_labels_for_samples = self.label_encoder.inverse_transform(self.y_test[test_idx])

        for cname, feats in tqdm.auto.tqdm(self.top_features_per_class.items(), total=len(self.top_features_per_class)):
            t1 = time.time()
            subset = self.x_data_aug[feats[:5]].copy()
            subset["is_class"] = (self.y_data == cname)
            sns.pairplot(subset, hue="is_class", diag_kind="hist", plot_kws={"alpha":0.4, "rasterized": True}, corner = True)
            plt.suptitle(cname, y=1.02)
            plt.show()
            print(f"Time to plot was {time.time()-t1}s")
            
        t1 = time.time()
        subset = self.x_data_aug[self.top_features].copy()
        subset["Group"] = self.y_data
        g = sns.pairplot(subset, hue="Group",
                         hue_order = ["Group_1", "Group_2", "Group_3", "Group_4", "Group_5", "Group_6", "Group_7", "Group_8"],
                         corner = True,
                         diag_kind="auto", plot_kws={"alpha":0.4, "s" : 8, "rasterized" : True})

        sns.move_legend(g, "upper right", bbox_to_anchor=(0.95, 0.95), fontsize=28, markerscale=10)
        plt.setp(g._legend.get_title(), fontsize=16)

        print(f"Took {time.time()-t1}s to run")
        
        mean_abs_shap = {cname: np.abs(self.shap_values[:, :, i]).mean()*1000
                  for i, cname in enumerate(self.class_names)}
        print(mean_abs_shap)
        
    def R2_per_feature(self):
        """
        Shows how the R2 for each group changes with the addition of important features.

        Returns
        -------
        """
        baseline = self.model.predict(self.x_test).mean(axis=0)

        def subset_explained_variance(shap_values, model_output, feature_names, test, baseline):
            """
            shap_values: array (n_samples, n_features) for ONE class/output
            model_output: array (n_samples,) — the actual model predictions (or logits/probs)
            feature_names: list of feature names
            test: list of feature names to test
            """
            idx = [feature_names.index(f) for f in test]
            partial_pred = shap_values[:, idx].sum(axis=1) + baseline
            return r2_score(model_output, partial_pred)
        
        predictions_for_shap_samples = self.model.predict(self.samples_to_explain)
        full_reconstruction = self.shap_values.sum(axis=1) + baseline
        print(r2_score(predictions_for_shap_samples, full_reconstruction))  # should be ~1.0 (or very close)
        
        subset_explained_variance(self.shap_values[:, :, 1], predictions_for_shap_samples[:, 1], self.feature_names, ["Metabolite_002", "Metabolite_003"], baseline = baseline[1])

        def plot_feature_contribution(index, ax):
            """
            A function that plots the R2 of the model output against the number of features used to explain it.
            index : int
            The index of the group to be examined
            ax : matplotlib.axes.Axes
            The axes to plot on.
            """
            # The `chosen_class_index` needs to be passed to the function,
            # as it's defined outside the function's scope but modified in the loop.
            # Or, even better, `index` is already passed to the function and can be used directly.
            ranked_features = self.top_features_per_class[self.class_names[index]] # Use features ranked for the specific class

            r2_curve = []

            for k in range(1, len(ranked_features) + 1):
                subset = ranked_features[:k]
                # Pass the SHAP values, predictions, and baseline specifically for the chosen class
                r2_curve.append(subset_explained_variance(
                    self.shap_values[:, :, index],
                    predictions_for_shap_samples[:, index],
                    self.feature_names,
                    subset,
                    baseline = baseline[index]
                    ))

            ax.plot(range(1, len(ranked_features)+1), r2_curve, marker='o')
            ax.set_ylim(0, 1)
            ax.set_xlabel("Number of top features included")
            ax.set_ylabel("R² (variance of model output explained)")
            ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"R-squared curve for {self.class_names[index]}") # Add a title for each subplot

            # Add labels to the points
            for k, r2_val in enumerate(r2_curve):
                feature_name = ranked_features[k]
                ax.annotate(feature_name, (k + 1, r2_val), textcoords="offset points", xytext=(5,-5), ha='left', fontsize=8)

            fig, axs = plt.subplots(self.num_groups//2, 2, figsize = (10, 16))

            for i in range(0, self.num_groups):
                row = i // 2
                col = i % 2
                ax = axs[row, col]
                plot_feature_contribution(i, ax)

            plt.suptitle("A set of graphs showing the improvement of the R2 on the model when more features are added per group")
            plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
            plt.show()
            
if __name__ == "__main__":
    #plt.close("all")
    DNN = DNN_for_Metabolites()
# Preprocessing, only need to run once
    DNN.generate_data()
    DNN.encode_labels()
    DNN.plot_seeded_values()
    DNN.preclustering_for_MI()         #<---- Takes a long time
    DNN.Mutual_Information_sweep()
    DNN.suggest_cluster_threshold()
    DNN.cluster(threshold = 0.999)
    DNN.add_features()
    DNN.finish_preprocessing()
    DNN.RF_on_created_columns()
# Model_training
    DNN.create_model()
    DNN.train_model()                  #<---- Takes a long time
    DNN.model_accuracy()
    DNN.observing_results()
    DNN.ROC_curves()
    DNN.calc_SHAP()
    DNN.SHAP_plots()
    DNN.SHAP_important_features()
    DNN.top_metabolite_pairplots()    #<---- Takes a long time
    DNN.R2_per_feature()
    
