# omics_data_analysis
A collection of python scripts that explore using Machine Learning to analyse -omic datasets


This repository contains 2 functions for the purposes of analysing -omic datasets using supervised and unsupervised machine learning methods. 

The supervised learning method is found in the function Random_Forest_Analysis. This uses a Random Forest Classifier that is trained and tested on -omic data. For effective classifying, parameter tuning is also included. Partial Dependence Plots are created for significant features with low correlation to try and observe the underlying interactions between the features.

The unsupervised learning method is found in the function SpectralBiclustering. This uses Spectral Biclustering to cluster a dataset by the features and samples. This allows the creation of hypothesis pathways between the features, that could suggest biological results if the sample clusters reflect the true groups of the data. 

These have been created in python primarily using the Scikit-Learn packages. I have created Notebooks that can be opened in Google Colab and Jupyter to allow a simple demonstration on the sample dataset.

The sup_func folder contains suplementary functions. Plot_Model_Results works as a visual alternative to a confusion matrix, showing the true and classified groups as the shape and colour of the points, respectively. load_human_metabolites is an example of the form of the function necessary to load a dataset. As datasets come in different forms, no standard function has been created to read in, for example, an excel file, as often there is additional information in the opening lines to be skipped and other issues. Therefore, the functions have been created to read in a data-loading function, that outputs a pandas dataframe with the sample names in the index, the feature names as the column names, with a column named "Group" containing the true groups. Alternatively, the function can return an x dataframe of the feature values and a y numpy array containing the groups. load_human_metabolites is an example of the first method.

Data contains a sample metabolomics dataset for the use in the example code. The data comes from this study: https://www.ebi.ac.uk/metabolights/editor/MTBLS1/overview [1]


[1] (Salek, R. and Griffin, J.), (2025) A metabolomic study of urinary changes in type 2 diabetes in human compared to the control group.
Available at: https://www.ebi.ac.uk/metabolights/editor/MTBLS1/overview (Accessed: 28 August 2026).
