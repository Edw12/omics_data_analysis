# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:12:56 2026

@author: Edwin
"""
from load_human_metabolites import load_human_metabolites
from Plot_Model_Results import Plot_Model_Results

from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay 

from itertools import combinations, cycle
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import numbers

import time

class Random_Forest_Analysis:
    
    def __init__(self, data_func, data_frame = False, random_state = 42):
        """
        Read in the dataset using a function that loads the data. Either, outputs the variables as x, the 
        true groups as y, and then other irrelevant info, or it returns a pandas data frame with the true
        groups in a column called "Group", in which case data_frame = True

        Parameters
        ----------
        data_func : function
            Function that loads the data as described above.
        data_frame : bool, optional
            If the function returns a df containing all information = True, otherwise = False. The default is False.

        Returns
        -------
        Saves necissary information
        """
        if data_frame == True:
            data = data_func()
            x = data.drop("Group", axis = 1)
            self.y = np.array(data["Group"].values)
        else:
            x, self.y, *_ = data_func()
            
        self.x = x.apply(pd.to_numeric, errors='coerce')
        self.random_state = random_state
        
    def generate_classifier(self, test_size = 0.3, val_size = 0.3, runtime = True, 
                            out = False, parameters = None):
        """
        Splits the data into train, test and validation sets. Trains the classifier on the train set and 
        generates predictions on the test set.
        
        Parameters
        ---------
        test_size : float, optional
            The size of the test data from the original dataset, so test data size = test_size*total data size
        val_size : float, optional
            The size of the validation data from the train data, so val data size = val_size*(1-test_size)*total data size
            The size of the train data = (1-val_size)*(1-test_size)*total_size
        runtime : bool, optional
            Prints the runtime. Default is True.
        out : bool, optional
            If true, returns the number of correctly allocated values and the number of test values. Default is false
        parameters : dict, optional
            .The parameters found using tune_hyperparams for the classifier. Most parameters must be integers,
            except for max_features, where it is often a float (0.5), but can be a string ("log2") 
        Returns
        -------
        None
        """
        t1 = time.time()
        
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.x, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state
            ) 

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train,
            test_size = val_size,
            stratify = y_train,
            random_state = self.random_state)

        if parameters == None:  # fine tune the parameters to avoid overfitting
            parameters = {
                'n_estimators' : 100,
                'max_depth': 10,
                'min_samples_leaf': 2,  
                'min_samples_split': 2,  # Minimum samples to split a node
                'max_features': "log2",  # How many features considered per split
                'ccp_alpha': 0.01  # Prunes branches that are overly complicated (more complicated without improving validation)
                }
        # No. of splits can't be > than no. of members of smallest class, issue for small datasets
        n_per_class = [] 
        name_class = []
        for a in np.unique(self.y_train):
            no = np.sum(self.y_train == a)
            n_per_class.append(no)
            name_class.append(a)
        index = np.argmin(np.array(n_per_class))
        n_splits = n_per_class[index]  # If no. members in smallest class is < 5, sets number of splits to this
        if n_splits > 5:  # n_splits > 5 can be excessive
            n_splits = 5
        if n_splits == 1:  # can't have a number of splits less than 2
            n_splits = 2
        print(f"The number of splits was {n_splits}, the smallest class was {name_class[index]}")
        self.n_splits = n_splits
        
        self.classifier = RandomForestClassifier(
            **parameters,
            class_weight="balanced",
            random_state=self.random_state
            )   
        self.classifier.fit(self.X_train, self.y_train)
        
        self.y_pred = self.classifier.predict(self.X_test)
        
        if runtime == True:
            print(f"generate_classifier took {time.time()-t1:.3f}s to run.")
            
        if out == True:
            return np.sum(self.y_pred==self.y_test), len(self.y_test)
    
    def tune_hyperparams(self, test_size = 0.3, val_size = 0.3, 
                         n_seeds = 3, n_iter = 100,
                         param_grid = None, runtime = True, out = False):
        """
        Performs a RandomizedSearchCV using different hyperparameters and random states to find the best
        parameters for building the classifier. Prints the average best results for use. The test_size, val_size
        should be kept the same as for generate_classifier for most useful results.
        This is computationally expensive and takes a long time to run, so it best used once to get the parameters
        and then not again.

        Parameters
        ----------
        test_size : float, optional
            First split for the test train. Refer to generate_classifier The default is 0.3.
        val_size : float, optional
            Second split. The default is 0.3.
        n_seeds : int, optional
            The number of different seeds to be used when testing the best hyperparameters. Having a high
            number of seeds and lots of tuning parameters can be computationally expensive and often unnecissary,
            however, 3 is too low to get a good idea. 10 is more appropriate. The default is 3.
        n_iter : int, optional
            The number of iterations of the RandomizedSearchCV, this is the number of parameter combinations on
            the grid that it selects. Higher numbers take longer, check more options, but are often unnecissary.
            The default is 100.
        param_grid : dictionary, optional
            Dictionary of hyperparameters to try for RandomizedSearchCV. There is a default dict if left as None. The default is None.
        runtime : bool, optional
            Prints runtime length. The default is True.
        out : bool, optional
            Returns a dictionary of the best parameters per run. The default is False

        Returns
        -------
        Prints the average results for the parameter scores to then be put into the main classifier.
        If out == True, returns a dictionary of the values for the best parameters for each random state.
        """
        

        def summarise_best_params(params_best):
            """
            Generalized averaging/summary of best_params_ dicts across multiple 
            RandomizedSearchCV runs. Works for any set of hyperparameter keys.
            
            Numeric parameters (int/float) are averaged.
            Non-numeric or mixed-type parameters (e.g. max_features: 'sqrt' vs 0.5) 
            are summarized as a frequency count instead.
            """
            collected = defaultdict(list)
            best_vals = {}
            for b in params_best:
                for key, value in b.items():
                    collected[key].append(value)
            print("--- Results of Hyperparameter testing ---")
            print("Averages / summaries across runs:")
            for key, values in collected.items():
                
                if all(isinstance(v, bool) for v in values):  # Checks whether they're all booleans
                    counts = Counter(values)
                    most_common, freq = counts.most_common(1)[0]
                    best_vals[key] = most_common
                    print(f"  {key} : mode = {most_common} ({freq}/{len(values)} runs)")
                    continue
                # Try to treat every value as numeric
                try:
                    numeric_values = [float(v) for v in values]
                    avg = sum(numeric_values) / len(numeric_values)
                    all_int = all(isinstance(v, numbers.Integral) for v in values)  # Checks whether all of the values are integers
                    if all_int == True:    
                        best_vals[key] = round(avg)  # Rounds to usable integer
                    else:
                        best_vals[key] = avg  # Keeps as true float value
                    print(f"  {key} : {avg:.4g}  (numeric mean, from {values})")
                except (TypeError, ValueError):
                    # Fall back to categorical: count occurrences as strings
                    counts = Counter(values)
                    most_common, freq = counts.most_common(1)[0]
                    if most_common is None:  # Prevents None being turned into a string "None"
                        best_vals[key] = None
                    try:
                        best_vals[key] = float(most_common)
                    except ValueError:
                        best_vals[key] = most_common
                    print(f"  {key} : mode = {most_common} ({freq}/{len(values)} runs)  full distribution = {dict(counts)}")

            return collected, best_vals
        
        random_gen = np.random.default_rng(seed = self.random_state)
        random_states = random_gen.integers(low = 0, high = 100, size = n_seeds)  # Set of random seeds for testing
        res = []
        params_best = []
        for i in random_states:
            # Use RandomizedSearchCV rather than GridSearchCV for lighter computation
                t1 = time.time()
                X_train, self.X_test, y_train, self.y_test = train_test_split(
                    self.x, self.y,
                    test_size=test_size,
                    stratify=self.y,
                    random_state=i
                    ) 
            
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_train, y_train,
                    test_size = val_size,
                    stratify = y_train,
                    random_state = i)
                
                n_per_class = []  # Number of splits cannot be greater than the number of members in each class
                name_class = []
                for a in np.unique(self.y_train):
                    no = np.sum(self.y_train == a)
                    n_per_class.append(no)
                    name_class.append(a)
                index = np.argmin(np.array(n_per_class))
                n_splits = n_per_class[index]
                if n_splits > 5:
                    n_splits = 5
                if n_splits == 1:
                    n_splits = 2

                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state= i,
                    class_weight="balanced"
                    )

                if param_grid == None:  # fine tune the parameters to avoid overfitting
                    param_grid = {  # Sample grid
                        'n_estimators' : [100, 200, 300, 400],
                        'max_depth': [3, 5, 7, 10],
                        'min_samples_leaf': [1, 2, 5, 10, 20],  
                        'min_samples_split': [2, 5, 10, 20],  # Minimum samples to split a node
                        'max_features': ['sqrt', 'log2', 0.5],  # How many features considered per split
                        'ccp_alpha': [0.0, 0.001, 0.01, 0.05]  # Prunes branches that are overly complicated (more complicated without improving validation)
                        }
            
                
        
        
                random_search = RandomizedSearchCV(  # Randomly selects n_iter from the grid of the parameter grid
                                           self.classifier,
                                           param_distributions=param_grid,
                                           n_iter=n_iter,  # you choose how many combos to try, regardless of grid size
                                           cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i),
                                           scoring='f1_macro',
                                           random_state=i,
                                           n_jobs=-1
                                           )
                random_search.fit(self.X_train, self.y_train)
    
                self.classifier = random_search.best_estimator_  # Usable, fitted model
        
                self.y_pred = self.classifier.predict(self.X_test)
        
                if runtime == True:
                    print(f"{np.where(random_states == i)[0]}: Random state {i} took {time.time()-t1:.3f}s to run.")
            
                res.append([i, np.sum(self.y_pred==self.y_test), len(self.y_test)])
                params_best.append(random_search.best_params_)
        for a in res:
            print(f"For random state {a[0]}, the correctly allocated fraction was {a[1]}/{a[2]} \n")
        print(f"The average success was {np.mean(a[1]/a[2])*100:.3f}%")
        collected, best_vals = summarise_best_params(params_best)
        
        if out == True:
            return collected, best_vals
            
        
    def classifier_scores(self, CM = True, bal_acc_sco = True, 
                          per_tes_sco = True, n_permutations = 200,
                          out = False, pri = True, runtime = True):
        """
        Scores for testing the effectiveness of the classifier. Uses the balanced accuracy score and the 
        permutation test score. Also generates a confusion matrix

        Parameters
        ----------
        CM : bool, optional
            Can generate a Confusion Matrix. The default is True.
        bal_acc_sco : bool, optional
            Can calculate the balanced accuracy score. The default is True.
        per_tes_sco : bool, optional
            Can calculate the permutation test score. Returns the score, the scores from each permutation
            and the p value. The default is True.
        n_permutations : int, optional
            The number of permutations for per_tes_sco. The default is 200.
        out : bool, optional
            If true, creates a list of the desired results and returns this. The default is False.
        pri : bool, optional
            If true, prints the desired results. The default is True.
        runtime : bool, optional
            Times the method, only relevant if per_tes_sco = True. The default is True.

        Returns
        -------
        list
            A list of the calculated values.
        """
        string = ""
        output = []
        if runtime == True:
            t1 = time.time()
        if bal_acc_sco == True:
            bas = balanced_accuracy_score(self.y_test, self.y_pred)
            string += f"The balanced accuracy score was: {bas} \n"
            output.append(bas)
        
        if CM == True:
            fig, ax = plt.subplots(figsize = (16, 10))
            ConfusionMatrixDisplay.from_predictions(self.y_test, self.y_pred, ax = ax)
        
        if per_tes_sco == True:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state= self.random_state)

            score, perm_scores, pvalue = permutation_test_score(
                self.classifier, self.X_train, self.y_train, 
                scoring="accuracy", cv=cv, 
                n_permutations=n_permutations, n_jobs=3, random_state= self.random_state  # n_permutations = 500 would be better
                )
            
            string += f"The permutation score was {score} and the p value for the permutations was {pvalue} \n"
            output.append([score, perm_scores, pvalue])
        
        if runtime == True:
            print(f"The runtime of classifier_scores is {int((time.time()-t1)//60)} minutes and {(time.time()-t1)%60:.3f} seconds")
            
        if pri == True:
            print(string)
        if out == True:
            return output
    
    def Plot_Results(self, markers = None, cmap_name = "tab10"):
        """
        Plots the results of the classifier.
        Parmeters
        ---------
        markers : list, optional
            A list of markers to be used when plotting. If you have >12 groups, you will need a list of markers
            equal to the number of groups. Otherwise, leave as none for the default list
        cmap_name : string, optional
            The name of a matplotlib colourmap, which the colours for the points will be taken from. Default is
            "tab10"
        Returns
        -------
        None
        """
        def build_colour_map(labels, cmap_name):
            cmap = plt.get_cmap(cmap_name, len(labels))
            return {label: cmap(i) for i, label in enumerate(labels)}
        
        def build_marker_map(labels, markers):
            if markers == None:
                # A default set of distinct matplotlib markers
                markers = ['s', '*', '^', 'h', 'v', 'P', 'X', '*', 'p', 'D', '<', '>']
    
            marker_cycle = cycle(markers)
            return {label: next(marker_cycle) for label in labels}

        labels = np.unique(self.y)
        colour_map = build_colour_map(labels, cmap_name = cmap_name)
        marker_map = build_marker_map(labels, markers = markers)
        Plot_Model_Results(X_test = self.X_test, y_test = self.y_test, y_pred = self.y_pred, 
                           colour_map = colour_map, marker_map = marker_map)
        
    def find_important(self, pri = True, out = False, scoring = "roc_auc_ovr", use_val = True, runtime = True):
        """
        Finds the important columns and prints them. If there are no significant features in the top 30, the
        partial dependance plots will not be plotted properly.

        For smaller datasets, the scoring algorithm can struggle to find significant variables, so other scorers 
        may want to be tried. It was found, for a small dataset that ROC AUC one vs rest, balanced_accuracy, f1_macro, 
        f1_weighted and matthews_corrcoef had limited success.
        
        Parameters
        ----------
        pri : bool, optional
            If True, prints the number of significant metabolites in the total dataset and the top 30 
            most important. The default = True
        out : bool, optional
            If true, returns the 30 most important metabolites. The default is False.
        scoring : string, optional
            Passed to the permutation importance, to score the importance. Only use multiclass scorers. Default is roc_auc_ovr
        use_val : bool optional
            If true, uses the validation set, otherwise uses whole withheld dataset. If your validation set is too
            small, the significance can always be zero. Therefore, merges the test and val data to validate
        Returns
        -------
        pandas_df
            The most important metabolites
        """
        # A more useful measure of importance for highly correlated data, like metabolomics
        # "involves randomly shuffling the values of a single feature and observing the resulting degradation of the model’s score" 
        # -- https://scikit-learn.org/stable/modules/permutation_importance.html
        
        t1 = time.time()
        
        if use_val == True:
            x = self.X_val
            y = self.y_val
        else:
            x = pd.concat([self.X_val, self.X_test])
            y = np.concatenate([self.y_val, self.y_test])
        
        perm = permutation_importance(
            self.classifier, X = x, y = y,
            n_repeats=20, random_state= self.random_state, n_jobs=3, scoring = scoring)
    
        
        importance_df = pd.DataFrame({
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
        }, index=self.x.columns)
        
        importance_df["lower_bound"] = importance_df["importance_mean"] - 2 * importance_df["importance_std"]
        importance_df["significant"] = importance_df["lower_bound"] > 0
        
        importance_df = importance_df.sort_values("importance_mean", ascending=False)

        # Top 30 most important
        top_30 = importance_df.head(30)
        print(f"The top 30 features: \n{top_30}")
        
        if pri == True:
            n_significant = importance_df["significant"].sum()
            print(f"{n_significant} of {len(importance_df)} features clear the mean-2*std>0 bar")
            # How many of the top 30 are actually significant?
            n_sig_in_top30 = top_30["significant"].sum()
            print(f"{n_sig_in_top30} of the top 30 are significant")
        
        
        # Significant features for later use
        self.importance_df = importance_df
        self.important_columns = importance_df[importance_df["significant"]].index
        
        if runtime == True:
            print(f"find_important took {int((time.time()-t1)//60)} minutes and {(time.time()-t1)%60:.3f}s to run")
        
        if out == True:
            return top_30

    def MI_matrix(self, pri = True, out = False, runtime = True):
        """
        A Mutual Information matrix meausures the dependance between 2 variables, = 0 if they are independent.
        Requires find_important
        Parameters
        ----------
        pri : bool, optional
            If true, prints the matrix for the top 30 most important metabolites. The default is True.
        out : bool, optional
            If true, returns the matrix for the top 30 most important metabolites. The default is False.
        runtime : bool, optional
            If true, prints the runtime of MI_matrix. The default is True.

        Returns
        -------
        pandas df
            See out
        """
        # Measure the dependancy between variables, 0 means independant
        
        t1 = time.time()
        
        mi_matrix = pd.DataFrame(0.0, index=self.important_columns, columns=self.important_columns)

        for col_a, col_b in combinations(self.important_columns, 2):
                score = mutual_info_regression(
                    self.x[[col_a]], self.x[col_b], random_state = self.random_state
                )[0]
                mi_matrix.loc[col_a, col_b] = score
                mi_matrix.loc[col_b, col_a] = score  # symmetric by construction now

        top_30 = mi_matrix
        self.mi_matrix = mi_matrix
        if runtime == True:
            print(f"The runtime of MI_matrix is {int((time.time()-t1)//60)} minutes and {(time.time()-t1)%60:.3f} seconds")
        if pri == True:
            print(top_30)
        if out == True:
            return top_30
        
    def partial_dependance_plots(self, target = "Diabetic_Female", pri_pairs = False, ret_pairs = False, 
                                 start = 0, n = 4, runtime = True):
        """
        Generates Partial Dependence plots for the pairs with high combined importance excluding the highest 
        MI scores (most independent) from the most important metabolites. There must be n pairs of significant
        values. This can throw an error if significant features can't be found, as the pairs list will be empty.

        Parameters
        ----------
        pri_pairs : bool, optional
            If true, prints the list of pairs. The default is False.
        ret_pairs : bool, optional
            If true, returns the list of pairs. The default is False.
        n : int, optional
            The number of pairs to be plotted. Default = 4
        start : int, optional
            The 1st pair to be used of the most interesting plotting pairs. Change to see different 
            sets of features compared. Default = 0

        Returns
        -------
        list
            Pairs with the lowest MI.
        """
        
        t1 = time.time()
        end = start + n
        # Start from your top permutation-importance features (not MI-sorted)
        # Exclude pairs with high MI (i.e., redundant pairs)
        candidate_pairs = [
            (a, b) for a, b in combinations(self.important_columns, 2)
            if self.mi_matrix.loc[a, b] < self.mi_matrix.stack().quantile(0.75)  # below the 75th percentile of MI
            ]
        # Among the remaining, pick the two with the highest combined importance
        top_n_pairs = sorted(
            candidate_pairs,
            key=lambda pair: self.importance_df.loc[pair[0], "importance_mean"] + self.importance_df.loc[pair[1], "importance_mean"],
            reverse=True
            )[start:end]
        pairs = []
        
        for feat_1, feat_2 in top_n_pairs:
            pairs.append((feat_1, feat_2))
        
        rows = math.ceil(n / 2)
        fig, axs = plt.subplots(rows, 2, figsize=(12, 4.5*rows))
        axs_flat = np.atleast_1d(axs).ravel()        
        
        display = PartialDependenceDisplay.from_estimator(
            self.classifier, self.X_test, pairs,
            kind="average", target=target, ax=axs_flat[:n],
            percentiles=(0.1, 0.75)  # Stops individual large values creating enormous axis  
            )

        for i in range(n):
            feat_1, feat_2 = pairs[i]
            pdp_ax = axs_flat[i]
            pdp_ax.set_xscale("symlog")
            pdp_ax.set_yscale("symlog")
            pdp_ax.set_xlabel(f"{feat_1} (log concentration)")
            pdp_ax.set_ylabel(f"{feat_2} (log concentration)")
            pdp_ax.set_title(f"Partial dependence: {feat_1} vs {feat_2}\n(predicted probability of {target})")

            # Attach a colorbar to this subplot's contour set
            contour_set = display.contours_[i]
            cbar = fig.colorbar(contour_set, ax=pdp_ax)
            cbar.set_label(f"Predicted probability: {target}", rotation=270, labelpad=15)

        # hide any unused axes if n is odd
        for j in range(n, len(axs_flat)):
            axs_flat[j].set_visible(False)

        plt.tight_layout(pad=2.0, w_pad=3.0)
        plt.show()
            
        if runtime == True:
            print(f"PDPs for {target} took {time.time()-t1:.3f}s to run")
        plt.tight_layout()
        plt.show()
        if pri_pairs == True:
            print(f"Candidate pairs: \n {candidate_pairs} \n top_n_pairs: \n {top_n_pairs} \n pairs: \n {pairs}")
        if ret_pairs == True:
            return (pairs)

    def run_everything(self, params = None):
        """
        A simple function to run the whole code. It is recommended to create your own pipeline with the
        methods you want to use, which allows easier parameter setting.

        Parameters
        ----------
        params : list, optional
            A list of lists, containing the parameters to be used. If used, there must be
            4 contained lists. 
            params[0] = [out, n_seeds, param_grid, n_iter, test_size, val_size] for use in tuning hyperparameters
            params[1] = [best_vals, out, test_size, val_size] for use in generate classifier. Still must be present if using tuning hyperparameters
            params[2] = [CM, bal_acc_sco, per_tes_sco, n_permutations] for use in clasifier_scores
            params[3] = [target, n, pri_pairs] for use in partial_dependence_plots
            For the requirements for the parameters in the lists, refer to the method text.
            The default is None.

        Returns
        -------
        None
        """
        print("Tune the hyperparameters.")
        next_step = input("\nWould you like to run? y/n")
        if next_step == "y":
            try: 
                out, n_seeds, param_grid, n_iter, test_size, val_size = params[0]
                collected, best_vals = self.tune_hyperparams(out = out, n_seeds = n_seeds, param_grid = param_grid, n_iter = n_iter, test_size = test_size, val_size = val_size)
                self.generate_classifier(parameters = best_vals, out = out, test_size = test_size, val_size = val_size)
            except:
                collected, best_vals = self.tune_hyperparams()
                self.generate_classifier(parameters = best_vals, out = True)
        
        print("\nIf hyperparameters were not tuned, generate the classifier.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":        
            try: 
                best_vals, out, test_size, val_size = params[1]
                self.generate_classifier(parameters = best_vals, out = out, test_size = test_size, val_size = val_size)
            except:
                self.generate_classifier()
                
        print("\nScore the success of the classifier.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try: 
                CM, bal_acc_sco, per_tes_sco, n_permutations = params[2]
                self.classifier_scores(CM = CM, bal_acc_sco = bal_acc_sco, per_tes_sco = per_tes_sco, n_permutations = n_permutations)
            except:
                self.classifier_scores()
                
        print("\nPlot the results of the classifier.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
                self.Plot_Results()
        print("\nFind the important features and create a Mutual Information matrix.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try:
                self.find_important(use_val = False)
                self.MI_matrix()
            except:
                self.find_important(use_val = False)
                self.MI_matrix()

        print("\nCreates Partial Dependence Plots for high pair importance features.")
        next_step = input("\nWould you like to run? y/n")
        
        if next_step == "y":
            try:
                target, n, pri_pairs = params[3]
                self.partial_dependance_plots(target = target, n = n, pri_pairs = pri_pairs)
            except:
                self.partial_dependance_plots()

if __name__ == "__main__":
    
    plt.close("all")
    
    analysis = Random_Forest_Analysis(load_human_metabolites, data_frame = True, random_state=42)
    
    analysis.run_everything()
