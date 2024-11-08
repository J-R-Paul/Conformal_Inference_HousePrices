# %%
from utils import QuantileRegressorCV, load_prep_data, check_coverage, sort_y_values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mapie.quantile_regression import MapieQuantileRegressor
from sklearn.linear_model import LassoCV, QuantileRegressor
from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample
from skimpy import Skimpy
import os
import logging
import seaborn as sns
import pickle as pkl
from sklearn.preprocessing import PolynomialFeatures
from utils import load_prep_data, train_cal_test_split

# Ensure the logs directory exists
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(filename=os.path.join(log_dir, '02-Non-Adaptive.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# %%
# Load and prepare data
logging.info("Loading and preparing data")
df = load_prep_data(polynomial=True)

# Display initial data information
logging.info(f"Data columns: {df.columns}")
logging.info(f"First few rows of data: {df.head()}")

# Prepare features and target variable
X = df.drop("SalePrice", axis=1)
y = np.log(df["SalePrice"])

# Split data into training, calibration, and testing sets
logging.info("Splitting data into training, calibration, and testing sets")
X_train, X_cal, X_test, y_train, y_cal, y_test = train_cal_test_split(X, y, test_size=0.2, cal_size=0.2, random_state=42)
# train_ind, cal_ind, test_ind = np.split(
#     df.sample(frac=1, random_state=42).index,
#     [int(0.6 * len(df)), int(0.8 * len(df))]
# )
# X_train, X_cal, X_test = X.iloc[train_ind], X.iloc[cal_ind], X.iloc[test_ind]
# y_train, y_cal, y_test = y.iloc[train_ind], y.iloc[cal_ind], y.iloc[test_ind]

# Scale features
logging.info("Scaling features")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)

# Set style for clean plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# %%
# Adaptive
# ------------------------------------------------------------------------------
# Adaptive Normalised Score Functions 
# ------------------------------------------------------------------------------
from mapie.conformity_scores import ResidualNormalisedScore
from sklearn.linear_model import LassoCV
from lightgbm import LGBMRegressor

class PosEstim(LassoCV):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(
            X, np.log(np.maximum(y, np.full(y.shape, np.float64(1e-8))))
        )
        return self

    def predict(self, X):
        y_pred = super().predict(X)
        return np.exp(y_pred)


base_model = LassoCV()

base_model.fit(X_train, y_train)

residual_estimator = LassoCV()
residual_estimator.fit(X_res, np.abs(np.subtract(y_res, base_model.predict(X_res))))

# wrapped_residual_estimator = PosEstim().fit(
#     X_res, np.abs(np.subtract(y_res, base_model.predict(X_res)))
# )

mapie = MapieRegressor(base_model, cv="prefit", 
                        conformity_score=ResidualNormalisedScore(
                            residual_estimator=residual_estimator,
                            prefit=True
                        ))

mapie.fit(X_cal, y_cal)

preds, pis = mapie.predict(X_test, alpha=0.05)


# %%
y_test_sorted, preds_sorted, lower_sorted, upper_sorted = sort_y_values(y_test, preds, pis)

fig, ax = plt.subplots()
# ax.scatter(y_test_sorted, preds_sorted)
# Determine which points fall within the predicted interval
within_interval = (y_test_sorted >= lower_sorted) & (y_test_sorted <= upper_sorted)

# Plot points with colors based on whether they fall within the interval
ax.scatter(y_test_sorted[within_interval], preds_sorted[within_interval], label="Within interval", color="green")
ax.scatter(y_test_sorted[~within_interval], preds_sorted[~within_interval], label="Outside interval", color="red")

ax.errorbar(
    y_test_sorted,
    preds_sorted,
    yerr=np.vstack((preds_sorted - lower_sorted, upper_sorted - preds_sorted)),
    fmt="none",
    ecolor="green",
    capsize=0,
    alpha=0.3
)
ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', label="45 degree line")
plt.show()

# %%

upper_sorted- lower_sorted 





# ------------------------------------------------------------------------------
# Quantile Regression 
# ------------------------------------------------------------------------------
# Quantile lasso
# %%
mapie = MapieQuantileRegressor(QuantileRegressor(alpha=0.01), cv="split")

mapie.fit(X_train, y_train)


# %%
preds, pis = mapie.predict(X_test, alpha=0.05)

y_test_sorted, preds_sorted, lower_sorted, upper_sorted = sort_y_values(y_test, preds, pis)

fig, ax = plt.subplots()

# Determine which points fall within the predicted interval
within_interval = (y_test_sorted >= lower_sorted) & (y_test_sorted <= upper_sorted)

# Plot points with colors based on whether they fall within the interval
ax.scatter(y_test_sorted[within_interval], preds_sorted[within_interval], label="Within interval", color="green")
ax.scatter(y_test_sorted[~within_interval], preds_sorted[~within_interval], label="Outside interval", color="red")

# Plot the interval bounds
ax.scatter(y_test_sorted, lower_sorted, label="Lower bound", color="blue", marker="_")
ax.scatter(y_test_sorted, upper_sorted, label="Upper bound", color="blue", marker="_")

# Plot the 45-degree line
ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', label="45 degree line")

ax.legend()
plt.show()

# %%


