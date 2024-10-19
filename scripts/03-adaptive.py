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


df = load_prep_data()

# Choose the features for X
features = ["BedroomAbvGr"]  # You can add more features here if needed
X = df[features]
X = df.drop("SalePrice", axis=1)
y = np.log(df["SalePrice"])

# Split the data
train_ind, test_ind = np.split(df.sample(frac=1, random_state=42).index, [int(0.95*len(df))])

# Function to select data based on indices
def select_data(data, indices):
    return data.iloc[indices] if isinstance(data, pd.DataFrame) else data[indices]

X_train, X_test = [select_data(X, ind) for ind in (train_ind, test_ind)]
y_train, y_test = [select_data(y, ind) for ind in (train_ind, test_ind)]

# Reshape X if it's a single feature
if X.shape[1] == 1:
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
# Adaptive
# ------------------------------------------------------------------------------
# Adaptive Normalised Score Functions 
# ------------------------------------------------------------------------------
from mapie.conformity_scores import residual_conformity_scores
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

class PosEstim(LinearRegression):
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


base_model = 





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




# %%
