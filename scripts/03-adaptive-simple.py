# %%
import os
import logging
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor, MapieQuantileRegressor

from utils import load_prep_data, train_cal_test_split, prep_conditional_data, check_coverage, plot_conditional_intervals



# Only set up file logging if not running in interactive/notebook environment
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    # Ensure the logs directory exists
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir, '03-Adaptive-simple.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # Set up basic console logging for interactive sessions
    logging.basicConfig(level=logging.CRITICAL, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
# %%
df = load_prep_data(polynomial=False)
X, y, groups  = prep_conditional_data(df)
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, groups, test_size=0.3, random_state=123)

# %%
# Quantile Regression
q_estimator = QuantileRegressor(quantile=0.5, alpha=0)
mapie = MapieQuantileRegressor(q_estimator, method="quantile", cv='split', alpha=0.05)
mapie.fit(X_train, y_train, calib_size=0.3, random_state=26)
# %%
# Predictions
y_pred, y_pis = mapie.predict(X_test)

plot_conditional_intervals(y_test, y_pred, y_pis, group_test, title="Quantile Conformal Prediction Intervals")

# %%
# Adaptive
# ------------------------------------------------------------------------------
# Adaptive Normalised Score Functions 
# ------------------------------------------------------------------------------
from mapie.conformity_scores import ResidualNormalisedScore
base = LinearRegression(fit_intercept=False)
X_train_res, X_res_res, y_train_res, y_res_res = train_test_split(X_train, y_train, test_size=0.3, random_state=26)
base.fit(X_train_res, y_train_res)
residual_estimator = LinearRegression(fit_intercept=False)
residual_estimator.fit(X_res_res, np.abs(np.subtract(y_res_res, base.predict(X_res_res))))

mapie = MapieRegressor(base, cv='prefit',  
                       conformity_score=ResidualNormalisedScore(
                                        residual_estimator=residual_estimator, 
                                        prefit=True
                                        ))

mapie.fit(X_train, y_train, random_state=26)
y_pred, y_pis = mapie.predict(X_test, alpha=0.05)
# %%
plot_conditional_intervals(y_test, y_pred, y_pis, group_test, title="Adaptive Conformal Prediction Intervals")

# %%
