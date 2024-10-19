# %%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_prep_data, check_coverage

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score_v2

# %% 
df = load_prep_data(polynomial=True)

df.head()
df.columns
# %%

X = df.drop("SalePrice", axis=1)
# X = df["BedroomAbvGr"].to_numpy().reshape(-1, 1)
y = np.log(df["SalePrice"])

train_ind, cal_ind, test_ind = np.split(df.sample(frac=1, random_state=42).index, [int(0.6*len(df)), int(0.8*len(df))])
X_train, X_cal, X_test = X.iloc[train_ind], X.iloc[cal_ind], X.iloc[test_ind]
y_train, y_cal, y_test = y.iloc[train_ind], y.iloc[cal_ind], y.iloc[test_ind]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_cal = scaler.transform(X_cal)


# %%
# lr = LinearRegression()

# lr.fit(X_train, y_train)

# mapie = MapieRegressor(lr, cv="split")
# mapie.fit(X_cal, y_cal)
# preds, pis = mapie.predict(X_test, alpha=0.05)



# def plot_mapie(preds, pis, X, y):
#     # Sort X and corresponding predictions and intervals
#     sorted_indices = np.argsort(X.ravel())
#     X_sorted = X.ravel()[sorted_indices]
#     preds_sorted = preds[sorted_indices]
#     lower_sorted = pis[:, 0].ravel()[sorted_indices]
#     upper_sorted = pis[:, 1].ravel()[sorted_indices]
#     y_sorted = y[sorted_indices]

#     # Plot test data, predictions, and prediction intervals
#     fig, ax = plt.subplots()
#     ax.scatter(X_sorted, y_sorted, label="test data")
#     ax.scatter(X_sorted, preds_sorted, label="predictions", color="orange")
#     ax.fill_between(
#         X_sorted,
#         lower_sorted,
#         upper_sorted,
#         alpha=0.5,
#         label="prediction interval",
#         color="green"
#     )
#     ax.legend() 
#     plt.show()

# # Call the updated function
# plot_mapie(preds, pis, y_test, y_test)

# %%
lasso = LassoCV()
lasso.fit(X_train, y_train)

mapie = MapieRegressor(lasso, cv="prefit")
mapie.fit(X_cal, y_cal)
preds, pis = mapie.predict(X_test, alpha=0.05)
# %%
sorted_indices = np.argsort(y_test)
y_test.ravel()[sorted_indices]

# %% 
# Sort the values based on y_test
sorted_indices = np.argsort(y_test)
y_test_sorted = y_test.ravel()[sorted_indices]
preds_sorted = preds[sorted_indices]
pi_lower = pis[:, 0].ravel()[sorted_indices]
pi_upper = pis[:, 1].ravel()[sorted_indices]

# Plot predicted values against true values
plt.errorbar(
    y_test,            # True values on the x-axis
    preds,             # Predicted values on the y-axis
    yerr=[preds - pis[:, 0].ravel(), pis[:, 1].ravel() - preds],  # Asymmetrical error bars for confidence intervals
    fmt='o',           # 'o' marker for the predicted points
    ecolor='green',    # Color for the error bars
    alpha=0.5,         # Transparency for error bars
    label="confidence interval",
    capsize=3          # Add caps to the error bars for better visibility
)
# 45 degree line
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', label="45 degree line")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.legend()
plt.show()

# %%
print(f"Coverage rate: {regression_coverage_score_v2(y_test, pis)}")
# %%


# Check coverage by group
groups_list = pd.qcut(df['LotFrontage'], q=5, labels=False)
groups = np.unique(groups_list)
X_scaled = scaler.transform(X)
full_preds, full_pis = mapie.predict(X_scaled, alpha=0.05)

for group in groups:
    group_indices = np.where(groups_list == group)[0]

# Find the intersection between group_indices and test_ind
    test_group_indices = np.intersect1d(group_indices, test_ind)
    print(f"Coverage rate for group {group}: {regression_coverage_score_v2(y[test_group_indices], full_pis[test_group_indices])}")

