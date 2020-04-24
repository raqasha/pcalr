"""
Collection of analyzed regression methods.
"""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class PCALR(BaseEstimator):
    def __init__(self, n_components = 2):
        self.n_components = n_components

    def fit(self, X, y):
        self.pca = PCA(self.n_components).fit(X, y)

        self.model = LinearRegression()
        self.model.fit(self.pca.transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.pca.transform(X))
