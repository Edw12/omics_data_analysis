# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:48:02 2026

@author: Edwin
"""

import pandas as pd

def load_human_metabolites():
    """
    Returns the human metabolites diabetes data as a pandas dataframe. Each datapoint is indexed with the name of the sample,
    the metabolites are numbered in the column names. The final column is the group, either Diabetic_Male, Diabetic_Female, 
    Control_Male or Control_Femal

    Returns
    -------
    Pandas data frame
        Each column contains a float except the final group column, which contains a string..
    """
    
    
    file = pd.ExcelFile("human metabolites transposed.xlsx")
    df1 = pd.read_excel(file, "m_MTBLS1_metabolite_profiling_N")
    df2 = pd.read_excel(file, "metabolites")

    df_group = df1[['Factor Value[Gender]', 'Factor Value[Metabolic syndrome]']]
    df_meta = df2.drop(columns = ["chemical_shift"])  # Metabolite data

    # Create the 'group' DataFrame with patient IDs (from df_meta.columns) as its index
    group = pd.DataFrame(" ", index=df_meta.columns, columns=["Group"])

    mask_a = df_group['Factor Value[Metabolic syndrome]'] == "diabetes mellitus"
    mask_b = df_group['Factor Value[Gender]'] == "Female"

    # These masks (mask_a, mask_b) are boolean Series with integer indices (0 to 131),
    # corresponding to the rows in df_group and the columns (patient IDs) in df_meta.
    # We use these masks to select the relevant patient IDs from df_meta.columns
    # and then assign the group labels to the 'group' DataFrame using those patient IDs as indexers.

    # Get patient IDs for each group based on the boolean masks
    mask_d_f_indices = df_meta.columns[mask_a & mask_b]  # Patient IDs for Diabetic Female
    mask_d_m_indices = df_meta.columns[mask_a & ~mask_b] # Patient IDs for Diabetic Male
    mask_c_f_indices = df_meta.columns[~mask_a & mask_b] # Patient IDs for Control Female
    mask_c_m_indices = df_meta.columns[~mask_a & ~mask_b] # Patient IDs for Control Male

    # Assign group labels to the 'group' DataFrame using the patient IDs as index
    group.loc[mask_d_f_indices, "Group"] = "Diabetic_Female"
    group.loc[mask_d_m_indices, "Group"] = "Diabetic_Male"
    group.loc[mask_c_f_indices, "Group"] = "Control_Female"
    group.loc[mask_c_m_indices, "Group"] = "Control_Male"

    data_full = pd.concat([df_meta.T, group], axis = 1)  # Index is the datapoint name, grouped

    return data_full

if __name__ == "__main__":
    data = load_human_metabolites()
    print(data.drop("Group", axis = 1))