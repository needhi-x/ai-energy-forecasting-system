from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate(y_test, predictions):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return rmse, r2