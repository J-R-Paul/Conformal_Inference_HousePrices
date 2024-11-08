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

from utils import load_prep_data, check_coverage, plot_predictions_with_ci


# %%
# Load results
path = "../pickles/02-Non-Adaptive.pkl"
with open(path, "rb") as f:
    y_pred, y_pis, STRATEGIES = pkl.load(f)

# %%
squeezed_array = np.squeeze(y_pis['cv_plus'], axis=-1)
y_temp = np.ravel(y_pred['cv_plus'])
y_lower = y_temp - squeezed_array[:, 0] 
y_upper = squeezed_array[:, 1] - y_temp

# %%
len(y_pred)
# %%
# Create figure with subplots
fig, ax = plt.subplots(len(y_pred)+1 , 1, figsize=(12,30))
names = ["Naive", "CV+", "Jackknife", "Jackknife+", "Jackknife+AB", "Split"]
for strategy, axi, title in zip(STRATEGIES.keys(), ax, names):
    y_temp = np.ravel(y_pred[strategy])
    y_lower = y_temp - y_pis[strategy][:, 0, 0]
    y_upper = y_pis[strategy][:, 1, 0] - y_temp
    # Plot predictions with confidence intervals
    plot_predictions_with_ci(y_test, y_temp, y_lower, y_upper, ax=axi, title=title)


# plt.tight_layout()
plt.show()
# %%
def plot_predictions_with_ci(y_test, y_pred, ci_lower, ci_upper, ax=None, title="Predictions vs True Values"):
    """
    Plot predictions with confidence intervals, highlighting those that don't contain the true value.
    """
    if ax is None:
        ax = plt.gca()
    
    # Convert to numpy arrays if they aren't already
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Create scatter plot
    sns.scatterplot(x=y_test, y=y_pred, label="Predictions", ax=ax)
    
    # Create boolean mask for points where true value is outside CI
    outside_ci = (y_test < (y_pred - ci_lower)) | (y_test > (y_pred + ci_upper))
    
    # Calculate coverage and average interval length
    coverage = (1 - outside_ci.mean()) * 100  # Convert to percentage
    avg_interval_length = np.mean(ci_upper + ci_lower)
    
    # Plot CIs that contain true value (gray)
    ax.errorbar(y_test[~outside_ci], y_pred[~outside_ci], 
               yerr=[ci_lower[~outside_ci], ci_upper[~outside_ci]], 
               fmt='o', ecolor='gray', alpha=0.5, label='CI (True Value Inside)')
    
    # Plot CIs that don't contain true value (red)
    ax.errorbar(y_test[outside_ci], y_pred[outside_ci], 
               yerr=[ci_lower[outside_ci], ci_upper[outside_ci]], 
               fmt='o', ecolor='red', alpha=0.5, label='CI (True Value Outside)')
    
    # Add diagonal line for reference
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    
    # Add coverage and interval length annotation
    annotation_text = f'Coverage: {coverage:.1f}%\nAvg Interval Length: {avg_interval_length:.2f}'
    ax.annotate(annotation_text,
                xy=(0.98, 0.02),  # Position in axes coordinates
                xycoords='axes fraction',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                ha='right',  # Horizontal alignment
                va='bottom')  # Vertical alignment
    
    return ax
# %%
