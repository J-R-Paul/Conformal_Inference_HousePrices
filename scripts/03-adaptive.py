# %%
# Standard Library Imports
import logging
import os
import pickle as pkl
import sys

# Third-Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mapie.metrics import (
    regression_coverage_score,
    regression_mean_width_score
)
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Local Imports
from utils import (
    QuantileRegressorCV,
    check_coverage,
    load_prep_data,
    sort_y_values,
    train_cal_test_split
)

# Only set up file logging if not running in interactive/notebook environment
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    # Ensure the logs directory exists
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir, '03-Adaptive.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # Set up basic console logging for interactive sessions
    logging.basicConfig(level=logging.CRITICAL, 
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
# %% Adaptive
# ------------------------------------------------------------------------------
# Adaptive Normalised Score Functions 
# ------------------------------------------------------------------------------
