# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:02:52 2026

@author: Edwin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def Plot_Model_Results(X_test, y_test, y_pred, colour_map, marker_map):
  """
  A function to compare the predicted and true groups.
  X_test: test data
  Pandas dataframe
  y_test: true labels
  Numpy array
  y_pred: predicted labels
  Numpy array
  colour_map: A dictionary of the form "label": "colour"
  marker_map: A dictionary of the form "label": "marker"
  
  Plots the data on the 2 Principal Components found from X_test, coloured by the predicted labels and shaped by the true labels.
  """
  pca = PCA(n_components = 2, random_state = 42)

  X_t = pca.fit_transform(X_test)
  
  print('Explained variability per principal component: {}'.format(pca.explained_variance_ratio_))

  # Put `X_t` into DataFrame
  X_pca = pd.DataFrame(X_t,columns=["PC1","PC2"])

  x_plot = X_pca["PC1"]
  y_plot = X_pca["PC2"]

  fig, ax = plt.subplots(figsize=(7, 6))

  correct = y_test == y_pred
  
  print(f"The proportion of correct allocations is {np.sum(correct)}/{len(y_test)} \nwhich is {round(np.sum(correct)/len(y_test), 3)*100}%")

  for t_label, marker in marker_map.items():          # T label = "A", "B", "C" etc.
      mask = y_test == t_label
      colours = [colour_map[p] for p in y_pred[mask]]
      ax.scatter(
          x_plot[mask], y_plot[mask],
          c=colours,
          marker=marker,
          edgecolor="black",
          linewidth=0.5,
          s=70,
          alpha=0.85,
          label=None,  # legends built manually below
      )

  # ---------------------------------------------------------------
  # 4. Build two separate legends: one for color (predicted), one for shape (true)
  # ---------------------------------------------------------------
  from matplotlib.lines import Line2D

  color_handles = [
      Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
             markeredgecolor="black", markersize=9, label=p)
      for p, c in colour_map.items()
  ]
  shape_handles = [
      Line2D([0], [0], marker=m, linestyle="", markerfacecolor="grey",
             markeredgecolor="black", markersize=9, label=t)
      for t, m in marker_map.items()
  ]

  legend1 = ax.legend(handles=color_handles, title="Predicted group",
                       loc="upper left", bbox_to_anchor=(1.02, 1))
  ax.add_artist(legend1)  # keep this legend when adding the second one
  ax.legend(handles=shape_handles, title="True group",
            loc="upper left", bbox_to_anchor=(1.02, 0.55))

  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.set_title("True group (shape) vs Predicted group (colour) plotted on PCs")
  fig.tight_layout()

  #fig.savefig("scatter_true_vs_predicted.png", dpi=150, bbox_inches="tight")
  plt.show()
