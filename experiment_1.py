"""
Eksperyment 1

Dla przykładu porównanie pomiędzy jakością:
- regresji liniowej (LR),
- LR z dynamiczną redukcją do 2 atrybutów przez PCA (PCALR).

Eksperyment przedstawia zależność osiąganych przez nie jakości
względem kroku i wielkości porcji danych.
"""

import met, helper
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=3)

# Define processing parameters
chunk_sizes = [100, 500, 1000]  # number of training instances
steps = [100, 500, 1000]  # prediction time horizon in instances
methods = {"PCALR": met.PCALR(), "LR": LinearRegression()}

# Read data
data = np.array(pd.read_csv("params_with_output.csv").values[:, 1:])
X = data[:, :-1]
y = data[:, -1]

# Experimental loop
scores = np.zeros((len(chunk_sizes), len(steps), len(methods)))

for i, chunk_size in enumerate(chunk_sizes):
    for j, step in enumerate(steps):
        results = helper.test(chunk_size, step, X, y, methods)
        scores[i, j] = results

# Store scores
np.save("results/experiment_1", scores)
