# %%
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Tuple, List

from sklearn.model_selection import train_test_split

from utils import load_prep_data

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
    
def check_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Checks the coverage rate of the prediction interval.
    """
    coverage = np.sum((y >= lower) & (y <= upper)) / len(y)
    return coverage


# %%
# ------------------------------------------------------------------------------
# Fit and predict
# ------------------------------------------------------------------------------
# Prep data
df = load_prep_data(polynomial=False)
y = df["SalePrice"].to_numpy()
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

# %%

# ------------------------------------------------------------------------------
# Plot the prediction interval
# ------------------------------------------------------------------------------
lower_grid = np.repeat(lower, len(y_test))
upper_grid = np.repeat(upper, len(y_test))


mean_value = np.mean(y_train)

# Set a seaborn darkgrid theme with a custom palette
sns.set_theme(style="darkgrid")
palette = sns.color_palette("viridis", 3)

fig, ax = plt.subplots(figsize=(14, 7))

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
ax.set_ylabel("House Value ($)", fontsize=14)
ax.set_title("Test Data Prediction Interval", fontsize=18, weight='bold')

# Coverage annotation
ax.text(
    0.98,
    0.02,
    f"Coverage: {coverage:.2f}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=13,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

# Legend
ax.legend(loc="upper right", fontsize=12)

# Remove x-axis ticks if desired
ax.set_xticks([])

plt.tight_layout()
plt.savefig(plot_path + "mean_prediction_interval.png", dpi=300)
plt.show()

