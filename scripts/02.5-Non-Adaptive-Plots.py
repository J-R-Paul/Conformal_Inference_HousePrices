# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, QuantileRegressor, Lasso
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.subsample import Subsample
from mapie.metrics import regression_coverage_score_v2

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer

import pickle as pkl

from lightgbm import LGBMRegressor

from utils import load_prep_data, check_coverage, plot_predictions_with_ci, train_cal_test_split

# %% Load Data
df = load_prep_data(polynomial=True)



# Prepare features and target variable
X = df.drop("SalePrice", axis=1)
y = np.log(df["SalePrice"])

# Split data into training, calibration, and testing sets
X_train, X_cal, X_test, y_train, y_cal, y_test = train_cal_test_split(X, y, test_size=0.2, cal_size=0.2, random_state=42)

# Standard scalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)

# Set style for clean plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
# %% Load results
# 
path = "../pickles/02-Non-Adaptive.pkl"
with open(path, "rb") as f:
    y_pred, y_pis, STRATEGIES = pkl.load(f)

def clean_results(y_pred, y_pis, y_test, strategy):
    residual = y_pred[strategy] - (y_pis[strategy][:, 1, 0] + y_pis[strategy][:, 0, 0])/2
    outliers = np.where(np.abs(residual) > 0.5)
    y_pred_s = np.delete(y_pred[strategy], outliers)
    y_pis_s = np.delete(y_pis[strategy], outliers, axis=0)
    y_test_s = np.delete(y_test, outliers)
    
    # take upper and lower
    y_lower = y_pred_s - y_pis_s[:, 0, 0]
    y_upper = y_pis_s[:, 1, 0] - y_pred_s
    return y_pred_s, y_test_s, y_lower, y_upper

# %%
# Create figure with subplots
fig, ax = plt.subplots(len(y_pred) , 1, figsize=(12,30))
names = STRATEGIES.keys()
for strategy, axi, title in zip(STRATEGIES.keys(), ax, names):
    y_p_i, y_test_i, y_lower_i, y_upper_i = clean_results(y_pred, y_pis, y_test, strategy)
    # Plot predictions with confidence intervals
    plot_predictions_with_ci(y_test_i, y_p_i, y_lower_i, y_upper_i, ax=axi, title=title)


# plt.tight_layout()
plt.show()


# %%
