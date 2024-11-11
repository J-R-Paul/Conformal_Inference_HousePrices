# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, Lasso

from mapie.regression import MapieRegressor
from mapie.subsample import Subsample


import pickle as pkl

from utils import load_prep_data, train_cal_test_split

import logging
import os

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
# Run lasso cv, get lambda, and store it
logging.info("Running LassoCV to get alpha")
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
alpha = lasso_cv.alpha_
logging.info(f"Alpha from LassoCV: {alpha}")

# %%
# Define model
# logging.info("Defining and fitting WrapRegressor model")
# gbm = WrapRegressor(Lasso(alpha=alpha))

# # Fit and calibrate model
# gbm.fit(X_train, y_train)
# gbm.calibrate(X_cal, y_cal)

# %%
# logging.info("Predicting intervals and base predictions")
# base_intervals = gbm.predict_int(X_test, confidence=0.95)
# base_pred = gbm.predict(X_test)
# base_intervals, base_pred

# lower = base_pred - base_intervals[:, 0]
# upper = base_intervals[:, 1] - base_pred

# %%
# Define strategies - using a subset for cleaner comparison
logging.info("Defining strategies for MapieRegressor")
STRATEGIES = {
    "naive": dict(method="naive"),
    "split": dict(cv="split", method="base"),
    "jackknife": dict(method="base", cv=-1, n_jobs=-1),
    "jackknife_plus": dict(method="plus", cv=-1, n_jobs=-1),
    "cv_plus": dict(method="plus", cv=10, n_jobs=-1),
    "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50), n_jobs=-1),
}

# Fit models and get predictions
logging.info("Fitting models and getting predictions for each strategy")
y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    logging.info(f"Processing strategy: {strategy}")
    # if strategy == "split":
    #     # If split, we use the lass_cv model which is already fitted
    #     mapie = MapieRegressor(lasso_cv, cv="prefit", method="base")
    #     mapie.fit(X_cal, y_cal)
    #     y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)
    # else:
    logging.info(f"Combining test with calibration data for strategy: {strategy}")
    # Combine train with calibration data
    X_train_cal = np.vstack([X_train, X_cal])
    y_train_cal = np.concatenate([y_train, y_cal])
    mapie = MapieRegressor(Lasso(alpha=alpha), **params)
    mapie.fit(X_train_cal, y_train_cal)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)

# Pickle results
path = "../pickles/02-Non-Adaptive.pkl"
try:
    logging.info(f"Saving results to {path}")
    with open(path, "wb") as f:
        pkl.dump((y_pred, y_pis, STRATEGIES), f)
    logging.info(f"Results successfully saved to {path}")
except Exception as e:
    logging.error(f"An error occurred while saving the results: {e}")
    fallback_path = "../pickles/02-Non-Adaptive-fallback.pkl"
    try:
        logging.info(f"Saving results to fallback path {fallback_path}")
        with open(fallback_path, "wb") as f:
            pkl.dump((y_pred, y_pis, STRATEGIES), f)
        logging.info(f"Results successfully saved to fallback path {fallback_path}")
    except Exception as fallback_e:
        logging.error(f"An error occurred while saving the results to the fallback path: {fallback_e}")