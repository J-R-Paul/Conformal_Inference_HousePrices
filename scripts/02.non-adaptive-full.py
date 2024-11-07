# %%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.subsample import Subsample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from lightgbm import LGBMRegressor

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


# Set style for clean plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define models
lasso_model = Pipeline([
    ("lasso", LassoCV(cv=5, random_state=42))
])

# Fit lasso, extract alpha and use it for quantile regression
lasso_model.fit(X_train, y_train)
alpha_lasso = lasso_model["lasso"].alpha_

quantile_model = Pipeline([
    ("quantile", QuantileRegressor(solver="highs", alpha=alpha_lasso))
])

# %%

# Define strategies - using a subset for cleaner comparison
STRATEGIES = {
    "naive": dict(method="naive"),
    "jackknife": dict(method="base", cv=-1),
    "jackknife_plus": dict(method="plus", cv=-1),
    "cv_plus": dict(method="plus", cv=10),
    "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50)),
    # "conformalized_quantile_regression": dict(
    #     method="quantile", cv="split", alpha=0.05
    # )
}

# Fit models and get predictions
y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    print(f'Strategy {strategy}')
    if strategy == "conformalized_quantile_regression":
        mapie = MapieQuantileRegressor(quantile_model, **params)
        mapie.fit(X_train, y_train, random_state=1)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test)
    else:
        mapie = MapieRegressor(LGBMRegressor(verbose=-1), **params)
        mapie.fit(X_train, y_train)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)

# %%
def calculate_metrics(y_true, y_pred, y_pis):
    """Calculate coverage and average interval length."""
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]
    
    # Calculate coverage
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    # Calculate average interval length
    avg_length = np.mean(upper - lower)
    
    return coverage, avg_length

def plot_prediction_vs_true(y_true, y_pred, y_pis, ax, title=None):
    """Plot predictions vs true values with prediction intervals."""
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_lower_sorted = y_pis[sort_idx, 0, 0]
    y_upper_sorted = y_pis[sort_idx, 1, 0]
    
    # Plot prediction intervals
    ax.fill_between(
        range(len(y_true)),
        y_lower_sorted,
        y_upper_sorted,
        alpha=0.2,
        label='Prediction Interval'
    )
    
    # Plot true values and predictions
    ax.plot(range(len(y_true)), y_true_sorted, 'r--', label='True Values', alpha=0.7)
    ax.plot(range(len(y_true)), y_pred_sorted, 'b-', label='Predictions', linewidth=2)
    
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(True, alpha=0.3)
    if ax.is_last_row():
        ax.set_xlabel('Sample Index (sorted by true value)')
    if ax.is_first_col():
        ax.set_ylabel('Value')
    
    # Add legend only to the first plot
    if title == list(STRATEGIES.keys())[0].replace('_', ' ').title():
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate and display metrics
print("\nPerformance Metrics:")
print("-" * 50)
print(f"{'Strategy':<35} {'Coverage':>10} {'Avg Length':>12}")
print("-" * 50)
for strategy in STRATEGIES.keys():
    coverage, avg_length = calculate_metrics(
        y_test.ravel(),
        y_pred[strategy].ravel(),
        y_pis[strategy]
    )
    print(f"{strategy.replace('_', ' ').title():<35} {coverage:>10.3f} {avg_length:>12.3f}")

# Create subplot grid
n_strategies = len(STRATEGIES)
n_rows = (n_strategies + 1) // 2
fig, axs = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
fig.suptitle('Predictions vs True Values with Prediction Intervals', y=1.02)

# Flatten axs array for easier iteration
axs_flat = axs.ravel()

# Plot each strategy
for idx, (strategy, ax) in enumerate(zip(STRATEGIES.keys(), axs_flat)):
    plot_prediction_vs_true(
        y_test.ravel(),
        y_pred[strategy].ravel(),
        y_pis[strategy],
        ax=ax,
        title=strategy.replace('_', ' ').title()
    )

# Remove empty subplots if odd number of strategies
for idx in range(len(STRATEGIES), len(axs_flat)):
    fig.delaxes(axs_flat[idx])

plt.tight_layout()
plt.show()
# %%
