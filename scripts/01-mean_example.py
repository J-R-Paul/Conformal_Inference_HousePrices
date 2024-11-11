# %%
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from utils import load_prep_data, check_coverage

plot_path = "../plots/01_mean_example/"

class FullConformalMean:
    """
    Full Conformal Inference for estimating the mean of a variable.
    
    This class implements the full conformal inference algorithm to produce
    a prediction interval for the mean of a dataset. It considers all possible
    candidate y-values within a specified grid and includes them in the prediction
    set based on conformity scores.
    
    Attributes:
        data_ (np.ndarray): The fitted data.
        n_ (int): Number of data points.
        sum_ (float): Sum of the data points.
    """
    
    def __init__(self):
        """
        Initializes the FullConformalMean instance.
        """
        self.data_ = None
        self.n_ = 0
        self.sum_ = 0.0
    
    def fit(self, data):
        """
        Fits the model using the provided data.
        
        Parameters:
            data (np.ndarray or pd.Series): A single sample of data.
        
        Raises:
            ValueError: If data is empty or not a 1-dimensional array/series.
            TypeError: If data is not a numpy array or pandas Series.
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(data, pd.Series):
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise TypeError("Data should be a numpy array or pandas Series.")
        
        if data.ndim != 1:
            raise ValueError("Data should be a 1-dimensional array or series.")
        
        if len(data) == 0:
            raise ValueError("Data should not be empty.")
        
        self.data_ = data
        self.n_ = len(data)
        self.sum_ = np.sum(data)
    
    def predict(self, alpha: float, grid_size: int = 1000) -> Tuple[float, float]:
        """
        Predicts the interval for the mean with the given significance level using full conformal inference.
        
        Parameters:
            alpha (float): Significance level (0 < alpha < 1).
            grid_size (int): Number of points in the grid for candidate y-values.
        
        Returns:
            tuple: A tuple containing the lower and upper bounds of the prediction interval.
        
        Raises:
            ValueError: If alpha is not between 0 and 1, or if the model has not been fitted.
        """
        if self.data_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' with appropriate data before prediction.")
        
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        
        # Define the grid of candidate y-values
        data_min = np.min(self.data_)
        data_max = np.max(self.data_)
        data_range = data_max - data_min
        buffer = data_range * 0.5 if data_range > 0 else 1.0  # In case all data points are the same
        y_grid = np.linspace(data_min - buffer, data_max + buffer, grid_size)
        
        # Calculate the required rank
        k = math.ceil((self.n_ + 1) * (1 - alpha))
        
        # Initialize list to collect valid y's
        valid_ys = []
        
        # Iterate over the grid and check conformity
        for y in y_grid:
            mu_prime = (self.sum_ + y) / (self.n_ + 1)
            # Compute conformity scores for all data points
            residuals = np.abs(self.data_ - mu_prime)
            # Conformity score for y
            y_residual = abs(y - mu_prime)
            # Rank: number of residuals less than y_residual
            rank = np.sum(residuals < y_residual) + 1  # +1 for y's own residual
            if rank <= k:
                valid_ys.append(y)
        
        if not valid_ys:
            raise RuntimeError("No valid y-values found. Consider increasing the grid size or adjusting the buffer.")
        
        # Since valid_ys are sorted (as y_grid is sorted), find continuous intervals
        # Here, we assume the prediction set is a single interval
        lower = min(valid_ys)
        upper = max(valid_ys)
        
        return (lower, upper)
    
# Create plots for both methods
def plot_conformal_intervals(method_name: str, y_test: np.ndarray, lower: float, upper: float, mean_value: float, coverage: float, avg_interval_length: float) -> None:
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis", 3)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create prediction interval grid
    lower_grid = np.repeat(lower, len(y_test))
    upper_grid = np.repeat(upper, len(y_test))
    
    # Scatter plot
    ax.scatter(
        range(len(y_test)),
        y_test,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Prediction interval
    ax.fill_between(
        range(len(y_test)),
        lower_grid,
        upper_grid,
        color=palette[1],
        alpha=0.4,
        label="Prediction Interval"
    )
    
    # Mean line
    ax.axhline(
        y=mean_value,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean: {mean_value:.2f}"
    )
    
    # Labels and title
    ax.set_xlabel("Sample Index", fontsize=14)
    ax.set_ylabel("Log House Value ($)", fontsize=14)
    ax.set_title(f"{method_name} Prediction Interval", fontsize=18, weight='bold')
    
    # Coverage annotation
    annotation_text = f'Coverage: {100*coverage:.3f}%\nAvg Interval Length: {avg_interval_length:.2f}'
    ax.annotate(annotation_text,
                xy=(0.98, 0.02),
                xycoords='axes fraction',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                ha='right',
                va='bottom')
    
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f"{method_name.lower().replace(' ', '_')}_mean_prediction_interval.png", dpi=300)
    plt.show()

# %%
# ------------------------------------------------------------------------------
# Fit and predict
# ------------------------------------------------------------------------------
# Prep data
df = load_prep_data(polynomial=False)
y = np.log(df["SalePrice"].to_numpy())
y_train, y_test = train_test_split(y, test_size=0.3, random_state=123)

# %%
# Fit model
fcm = FullConformalMean()
fcm.fit(y_train)
# %% Predictions
lower, upper = fcm.predict(alpha=0.05)

coverage = check_coverage(lower, upper, y_test)

print(f"Mean: {np.mean(y_train)}")
print(f"Lower: {lower}")
print(f"Upper: {upper}")
print(f"Coverage on test data: {coverage}")
avg_interval_length= upper - lower
# %%
# Get qhat value
print(f"qhat: {np.abs(np.mean(y_train) -  lower)}")
print(f"Mean: {np.mean(y_train)}")

# %%
# Set a seaborn darkgrid theme with a custom palette
sns.set_theme(style="darkgrid")
palette = sns.color_palette("viridis", 3)

plot_conformal_intervals("Full Conformal", y_test, lower, upper, np.mean(y_train), coverage, avg_interval_length)



# %%
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

X_train = np.ones_like(y_train).reshape(-1, 1)
X_test = np.ones_like(y_test).reshape(-1, 1)



# %%
STRATEGIES = {
    "naive": dict(method="naive"),
    "split": dict(cv="split", method="base"),
    "jackknife": dict(method="base", cv=-1, n_jobs=-1),
    "jackknife_plus": dict(method="plus", cv=-1, n_jobs=-1),
    "cv_plus": dict(method="plus", cv=10, n_jobs=-1),
    "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50), n_jobs=-1),
}
y_pred = {}
y_pis = {}
for strategy, params in STRATEGIES.items():
    # Combine test with calibration data
    
    mapie = MapieRegressor(LinearRegression(fit_intercept=False), **params)
    mapie.fit(X_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)
# %%

def clean_results(y_pred, y_pis, y_test):
    # Returns lower: float, upper: float, mean_value: float, coverage: float, avg_interval_length: float
    lower = y_pis[:, 0, 0][0]
    upper = y_pis[:, 1, 0][0]
    mean_value = y_pred[0]
    coverage = check_coverage(lower, upper, y_test)
    avg_interval_length = upper - lower
    return lower, upper, mean_value, coverage, avg_interval_length

def clean_title(strategy: str) -> str:
    return strategy.replace("_plus", "+").replace("_", " ").replace("cv", "CV").title()

# %%
for strategy in STRATEGIES.keys():
    strat_clean = clean_title(strategy)
    lower, upper, mean_value, coverage, avg_interval_length = clean_results(y_pred[strategy], y_pis[strategy], y_test)
    plot_conformal_intervals(strat_clean, y_test, lower, upper, mean_value, coverage, avg_interval_length)
# %%
