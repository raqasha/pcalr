import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define hyperparameters
n_components = 20
chunk_size = 1100
n_splits = 5

# Read data
df = pd.read_csv("params_with_output.csv")
data = np.array(df.values[:,1:])
X = data[:,:-1]
y = data[:,-1]

# CV
kf = KFold(n_splits = n_splits, shuffle=True, random_state=1410)
scores = np.zeros(n_splits)
g_scores = []
i = 0

while (i+1)*chunk_size < X.shape[0]:
    _X = np.copy(X[i*chunk_size:(i+1)*chunk_size])
    _y = np.copy(y[i*chunk_size:(i+1)*chunk_size])

    for fold, (train, test) in enumerate(kf.split(_X)):
        # Normalization
        pca = PCA(n_components=n_components).fit(_X[train])

        __X = pca.transform(_X)
        #__X = np.copy(_X)

        # Training and testing
        est = LinearRegression()
        # est = MLPRegressor(solver="lbfgs", hidden_layer_sizes=(3,))

        est.fit(__X[train], _y[train])
        # print(est.coef_)

        # Current score
        y_pred = est.predict(__X[test])
        score = r2_score(_y[test], y_pred)

        scores[fold] = score

    g_scores.append(np.copy(scores))

    print("[%03i] %.3f (+-%.2f)" % (i, np.mean(scores), np.std(scores)))
    i+=1

g_scores = np.array(g_scores)

print("---\nGLOBAL %.3f (+-%.2f)" % (np.mean(g_scores), np.std(g_scores)))
