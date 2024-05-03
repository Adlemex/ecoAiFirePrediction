import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Perceptron, Lars
from sklearn.model_selection import train_test_split, cross_val_score

df_common = pd.read_csv("common.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df_common[
        ["latitude", "longitude", "acq_day", "acq_month", "temperature", "humidity",
         "vapour_pressure",
         "soil_moisture", "soil_temperature", "wind", "dew_point"]], df_common['frp'],
    test_size=0.33, random_state=42)
perceptron: linear_model.GammaRegressor = linear_model.GammaRegressor().fit(X_train, y_train)

area_fire_pred = perceptron.predict(X_test)

accuracy = perceptron.score(X_train, y_train)
print(accuracy)


def get_conf_int(alpha, lr, X=X_train, y=y_train):
    """
    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """

    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = X.copy()
    X_aux.insert(0, 'const', 1)
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - lr.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha / 2, dof)
    gap = t_val * np.sqrt(mse * var_params)

    return pd.DataFrame({
        'lower': coefs - gap, 'upper': coefs + gap
    }, index=X_aux.columns)


def predict(data):
    test = pd.DataFrame([data], columns=["latitude", "longitude", "acq_day", "acq_month", "temperature", "humidity",
                                         "vapour_pressure",
                                         "soil_moisture", "soil_temperature", "wind", "dew_point"])
    return perceptron.predict(test)[0]


alpha = 0.05
# fit a sklearn LinearRegression model
lin_model = LinearRegression().fit(X_train, y_train)

print(get_conf_int(alpha, lin_model, X_train, y_train))
