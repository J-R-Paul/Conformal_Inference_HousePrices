# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, QuantileRegressor
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.subsample import Subsample
from mapie.metrics import regression_coverage_score_v2

from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer



from lightgbm import LGBMRegressor



from utils import load_prep_data, check_coverage



# %%
# Load and prepare data
df = load_prep_data(polynomial=True)

# Display initial data information
df.head()
df.columns

# Prepare features and target variable
X = df.drop("SalePrice", axis=1)
y = np.log(df["SalePrice"])

# Split data into training, calibration, and testing sets
train_ind, cal_ind, test_ind = np.split(
    df.sample(frac=1, random_state=42).index,
    [int(0.6 * len(df)), int(0.8 * len(df))]
)
X_train, X_cal, X_test = X.iloc[train_ind], X.iloc[cal_ind], X.iloc[test_ind]
y_train, y_cal, y_test = y.iloc[train_ind], y.iloc[cal_ind], y.iloc[test_ind]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cal = scaler.transform(X_cal)
X_test = scaler.transform(X_test)

# Set style for clean plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %%
# Define model
gbm = WrapRegressor(LGBMRegressor(n_estimators=1000, random_state=2314, verbosity=-1))

# Fit model
gbm.fit(X_train, y_train)

gbm
# %%
gbm.calibrate(X_cal, y_cal)

# %%
intervals = gbm.predict_p(X_test, confidence=0.99)
intervals
# %%
gbm.evaluate(X_test, y_test)
# %%
