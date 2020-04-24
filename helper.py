import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score

def test(chunk_size, step, X, y, methods):
    """
    Base experimental loop for given processing parameters and
    method dictionary
    """
    # Get number of patterns
    n_samples = X.shape[0]

    # Prepare storage for all predictions
    predictions = np.zeros((len(methods), y.shape[0]))

    # Processing loop
    pointer = chunk_size

    while pointer < n_samples - step:
        # Establish training range
        start, end = pointer - chunk_size, pointer
        # print(start, end)

        # Fit models
        for idx, method in enumerate(methods):
            model = clone(methods[method]).fit(X[start:end], y[start:end])

            # Estimate prediction
            predictions[idx, end : end + step] = model.predict(X[end : end + step])

        # Iterate
        pointer += step

    # Calculate and return achieved score
    scores = []
    for idx, method in enumerate(methods):
        scores.append(r2_score(y[chunk_size:-step], predictions[idx, chunk_size:-step]))

    return np.array(scores)
