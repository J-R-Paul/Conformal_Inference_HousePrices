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
X = df.drop("SalePrice", axis=1)
# 123
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3, random_state=123)

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

ones_train = np.ones_like(y_train).reshape(-1, 1)
ones_test = np.ones_like(y_test).reshape(-1, 1)



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
    mapie.fit(ones_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(ones_test, alpha=0.05)
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
# Subset of variables with varying degree of coverage
def analyze_coverage_disparity(X_test: pd.DataFrame, y_test: np.ndarray, 
                            lower: float, upper: float) -> pd.DataFrame:
    """
    Analyzes coverage disparities across different features in X_test.
    
    Parameters:
        X_test: Test features DataFrame
        y_test: Test target values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
    
    Returns:
        DataFrame with coverage disparities for each feature
    """
    results = []
    
    # Analyze each feature
    for column in X_test.columns:
        # For numerical features, split at median
        if X_test[column].dtype in ['int64', 'float64']:
            median = X_test[column].median()
            mask_below = X_test[column] <= median
            mask_above = X_test[column] > median
            groups = {
                'below_median': y_test[mask_below],
                'above_median': y_test[mask_above]
            }
        else:
            # For categorical variables, group by each unique value
            groups = {
                str(val): y_test[X_test[column] == val]
                for val in X_test[column].unique()
            }
        
        # Calculate coverage for each group
        coverages = {
            group: check_coverage(lower, upper, values) 
            for group, values in groups.items()
            if len(values) > 0  # Only calculate if group is non-empty
        }
        
        # Calculate disparity (max difference in coverage)
        if len(coverages) >= 2:  # Only calculate disparity if we have at least 2 groups
            max_coverage = max(coverages.values())
            min_coverage = min(coverages.values())
            disparity = max_coverage - min_coverage
            
            results.append({
                'feature': column,
                'coverage_disparity': disparity,
                'max_coverage': max_coverage,
                'min_coverage': min_coverage,
                'group_coverages': coverages
            })
    
    # Convert to DataFrame and sort by disparity
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coverage_disparity', ascending=False)
    
    return results_df

# Modify the data splitting to maintain DataFrame structure
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# Fit model
fcm = FullConformalMean()
fcm.fit(y_train)

# Get predictions
lower, upper = fcm.predict(alpha=0.05)

# Analyze coverage disparities
coverage_analysis = analyze_coverage_disparity(X_test, y_test, lower, upper)

# Print results
print("\nTop 5 Features with Largest Coverage Disparities:")
print("================================================")
for _, row in coverage_analysis.head().iterrows():
    print(f"\nFeature: {row['feature']}")
    print(f"Coverage Disparity: {row['coverage_disparity']:.3f}")
    print(f"Group Coverages: {row['group_coverages']}")

# %%
# Plot function remains the same
def plot_coverage_disparities(coverage_analysis: pd.DataFrame, top_n: int = 5):
    plt.figure(figsize=(12, 6))
    
    # Get top N features
    top_features = coverage_analysis.head(top_n)
    
    # Create bar plot
    bars = plt.bar(range(len(top_features)), top_features['coverage_disparity'])
    plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title('Top Features with Largest Coverage Disparities')
    plt.xlabel('Feature')
    plt.ylabel('Coverage Disparity')
    
    plt.tight_layout()
    plt.savefig(plot_path + "coverage_disparities.png", dpi=300)
    plt.show()

# Plot top 5 features with largest disparities
plot_coverage_disparities(coverage_analysis, top_n=5)

# %%
def plot_conformal_intervals_againts(x_var: str, method_name: str, y_test: np.ndarray, X_test: pd.DataFrame, 
                        lower: float, upper: float, mean_value: float, 
                        coverage: float, avg_interval_length: float) -> None:
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis", 3)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create prediction interval grid
    lower_grid = np.repeat(lower, len(y_test))
    upper_grid = np.repeat(upper, len(y_test))
    
    # Get GarageCars values
    x_values = X_test[x_var].values
    
    # Sort everything by GarageCars for better visualization
    sort_idx = np.argsort(x_values)
    x_values = x_values[sort_idx]
    y_sorted = y_test[sort_idx]
    lower_grid = lower_grid[sort_idx]
    upper_grid = upper_grid[sort_idx]
    
    # Scatter plot
    ax.scatter(
        x_values,
        y_sorted,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Prediction interval
    ax.fill_between(
        x_values,
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
    ax.set_xlabel("Overall Quality", fontsize=14)
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
    
    plt.tight_layout()
    plt.savefig(plot_path + f"{method_name.lower().replace(' ', '_')}_mean_prediction_interval_against_{x_var.replace(' ', '_')}.png", dpi=300)
    plt.show()

# Call the function with the additional X_test parameter
plot_conformal_intervals_againts("OverallQual", "Full Conformal", y_test, X_test, lower, upper, 
                        np.mean(y_train), coverage, avg_interval_length)
# %%
# 
class MondrianConformalMean:
    """
    Mondrian Conformal Inference for estimating the mean of a variable within groups.
    """
    
    def __init__(self):
        self.group_data_ = {}
        self.fitted_ = False
    
    def fit(self, data, groups):
        """
        Fits the model using the provided data and group information.
        
        Parameters:
            data (np.ndarray): The response variable data
            groups (np.ndarray): The grouping variable (OverallQual)
        """
        if len(data) != len(groups):
            raise ValueError("Data and groups must have the same length")
            
        # Store data by group
        unique_groups = np.unique(groups)
        for group in unique_groups:
            self.group_data_[group] = data[groups == group]
            
        self.fitted_ = True
    
    def predict(self, alpha: float, grid_size: int = 1000) -> dict:
        """
        Predicts intervals for each group.
        
        Returns:
            dict: Dictionary with group as key and (lower, upper) bounds as value
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        intervals = {}
        
        for group, data in self.group_data_.items():
            n = len(data)
            sum_data = np.sum(data)
            
            # Define grid for this group
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            buffer = data_range * 0.5 if data_range > 0 else 1.0
            y_grid = np.linspace(data_min - buffer, data_max + buffer, grid_size)
            
            # Calculate required rank for this group
            k = math.ceil((n + 1) * (1 - alpha))
            
            valid_ys = []
            for y in y_grid:
                mu_prime = (sum_data + y) / (n + 1)
                residuals = np.abs(data - mu_prime)
                y_residual = abs(y - mu_prime)
                rank = np.sum(residuals < y_residual) + 1
                if rank <= k:
                    valid_ys.append(y)
            
            if valid_ys:
                intervals[group] = (min(valid_ys), max(valid_ys))
            else:
                raise RuntimeError(f"No valid intervals found for group {group}")
                
        return intervals

def plot_mondrian_intervals(y_test, groups_test, intervals, mean_by_group):
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis", 3)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Sort everything by group for visualization
    sort_idx = np.argsort(groups_test)
    y_test_sorted = y_test[sort_idx]
    groups_test_sorted = groups_test[sort_idx]
    
    # Scatter plot
    ax.scatter(
        range(len(y_test)),
        y_test_sorted,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Plot intervals for each group
    current_idx = 0
    for group in sorted(intervals.keys()):
        group_mask = groups_test_sorted == group
        group_size = np.sum(group_mask)
        
        if group_size > 0:
            lower, upper = intervals[group]
            ax.fill_between(
                range(current_idx, current_idx + group_size),
                [lower] * group_size,
                [upper] * group_size,
                color=palette[1],
                alpha=0.4,
                label="Prediction Interval" if current_idx == 0 else ""
            )
            
            # Plot group mean
            ax.hlines(
                mean_by_group[group],
                current_idx,
                current_idx + group_size,
                colors='red',
                linestyles='--',
                label="Group Mean" if current_idx == 0 else ""
            )
            
            # Calculate coverage for annotation
            group_y = y_test[groups_test == group]
            coverage = np.mean((group_y >= lower) & (group_y <= upper))
            interval_length = upper - lower
            
            # Add annotation for this group
            mid_point = current_idx + group_size/2
            
            # Determine if annotation should be above or below
            if group in [3, 10]:  # Place these groups' labels below
                y_pos = lower
                vert_offset = -10  # Negative offset for below
                vert_align = 'top'
            else:  # Place other groups' labels above
                y_pos = upper
                vert_offset = 10  # Positive offset for above
                vert_align = 'bottom'
            
            ax.annotate(
                f'Group {group}\nCoverage: {coverage:.1%}\nLength: {interval_length:.2f}',
                xy=(mid_point, y_pos),
                xytext=(0, vert_offset),
                textcoords='offset points',
                ha='center',
                va=vert_align,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                fontsize=8
            )
            
            current_idx += group_size
    
    # Labels and title
    ax.set_xlabel("Sample Index (sorted by Overall Quality)", fontsize=14)
    ax.set_ylabel("Log House Value ($)", fontsize=14)
    ax.set_title("Mondrian Conformal Prediction Intervals by Overall Quality", fontsize=18, weight='bold')
    
    ax.legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plot_path + "mondrian_mean_prediction_interval.png", dpi=300)
    plt.show()

# %%
class MondrianConformalMean:
    """
    Mondrian Conformal Inference for estimating the mean of a variable within groups.
    """
    
    def __init__(self):
        self.group_data_ = {}
        self.fitted_ = False
        
    def _combine_groups(self, data, groups):
        """
        Helper method to combine specified groups.
        """
        new_groups = groups.copy()
        # Combine groups 1-4
        new_groups[groups <= 4] = 4
        # Combine groups 9-10
        new_groups[groups >= 9] = 9
        return new_groups
    
    def fit(self, data, groups):
        """
        Fits the model using the provided data and group information.
        
        Parameters:
            data (np.ndarray): The response variable data
            groups (np.ndarray): The grouping variable (OverallQual)
        """
        if len(data) != len(groups):
            raise ValueError("Data and groups must have the same length")
        
        # Combine groups
        combined_groups = self._combine_groups(data, groups)
            
        # Store data by group
        unique_groups = np.unique(combined_groups)
        for group in unique_groups:
            self.group_data_[group] = data[combined_groups == group]
            
        self.fitted_ = True
    
    def predict(self, alpha: float, grid_size: int = 1000) -> dict:
        """
        Predicts intervals for each group.
        
        Returns:
            dict: Dictionary with group as key and (lower, upper) bounds as value
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        intervals = {}
        
        for group, data in self.group_data_.items():
            n = len(data)
            sum_data = np.sum(data)
            
            # Define grid for this group
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            buffer = data_range * 0.5 if data_range > 0 else 1.0
            y_grid = np.linspace(data_min - buffer, data_max + buffer, grid_size)
            
            # Calculate required rank for this group
            k = math.ceil((n + 1) * (1 - alpha))
            
            valid_ys = []
            for y in y_grid:
                mu_prime = (sum_data + y) / (n + 1)
                residuals = np.abs(data - mu_prime)
                y_residual = abs(y - mu_prime)
                rank = np.sum(residuals < y_residual) + 1
                if rank <= k:
                    valid_ys.append(y)
            
            if valid_ys:
                intervals[group] = (min(valid_ys), max(valid_ys))
            else:
                raise RuntimeError(f"No valid intervals found for group {group}")
                
        return intervals

def plot_mondrian_intervals(y_test, groups_test, intervals, mean_by_group):
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis", 3)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Combine groups in test data
    groups_test_combined = groups_test.copy()
    groups_test_combined[groups_test <= 4] = 4
    groups_test_combined[groups_test >= 9] = 9
    
    # Sort everything by group for visualization
    sort_idx = np.argsort(groups_test_combined)
    y_test_sorted = y_test[sort_idx]
    groups_test_sorted = groups_test_combined[sort_idx]
    
    # Scatter plot
    ax.scatter(
        range(len(y_test)),
        y_test_sorted,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Plot intervals for each group
    current_idx = 0
    for group in sorted(intervals.keys()):
        group_mask = groups_test_sorted == group
        group_size = np.sum(group_mask)
        
        if group_size > 0:
            lower, upper = intervals[group]
            ax.fill_between(
                range(current_idx, current_idx + group_size),
                [lower] * group_size,
                [upper] * group_size,
                color=palette[1],
                alpha=0.4,
                label="Prediction Interval" if current_idx == 0 else ""
            )
            
            # Plot group mean
            ax.hlines(
                mean_by_group[group],
                current_idx,
                current_idx + group_size,
                colors='red',
                linestyles='--',
                label="Group Mean" if current_idx == 0 else ""
            )
            
            # Calculate coverage for annotation
            group_y = y_test[groups_test_combined == group]
            coverage = np.mean((group_y >= lower) & (group_y <= upper))
            interval_length = upper - lower
            
            # Add annotation for this group
            mid_point = current_idx + group_size/2
            
            # Determine label text based on group
            if group == 4:
                group_label = "Groups 1-4"
            elif group == 9:
                group_label = "Groups 9-10"
            else:
                group_label = f"Group {group}"
            
            # Determine if annotation should be above or below
            if group == 9:  # Place these groups' labels below
                y_pos = lower
                vert_offset = -10
                vert_align = 'top'
            else:  # Place other groups' labels above
                y_pos = upper
                vert_offset = 10
                vert_align = 'bottom'
            
            ax.annotate(
                f'{group_label}\nCoverage: {coverage:.1%}\nLength: {interval_length:.2f}',
                xy=(mid_point, y_pos),
                xytext=(0, vert_offset),
                textcoords='offset points',
                ha='center',
                va=vert_align,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                fontsize=8
            )
            
            current_idx += group_size
    
    # Labels and title
    ax.set_xlabel("Sample Index (sorted by Overall Quality)", fontsize=14)
    ax.set_ylabel("Log House Value ($)", fontsize=14)
    ax.set_title("Mondrian Conformal Prediction Intervals by Overall Quality", fontsize=18, weight='bold')
    
    ax.legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plot_path + "mondrian_mean_prediction_interval.png", dpi=300)
    plt.show()

# %%
class MondrianConformalMean:
    """
    Mondrian Conformal Inference for estimating the mean of a variable within groups.
    """
    
    def __init__(self):
        self.group_data_ = {}
        self.global_mean_ = None
        self.fitted_ = False
        
    def _combine_groups(self, data, groups):
        """
        Helper method to combine specified groups.
        """
        new_groups = groups.copy()
        # Combine groups 1-4
        new_groups[groups <= 4] = 4
        # Combine groups 9-10
        new_groups[groups >= 9] = 9
        return new_groups
    
    def fit(self, data, groups):
        """
        Fits the model using the provided data and group information.
        """
        if len(data) != len(groups):
            raise ValueError("Data and groups must have the same length")
        
        # Calculate global mean
        self.global_mean_ = np.mean(data)
        
        # Combine groups and store data
        combined_groups = self._combine_groups(data, groups)
        unique_groups = np.unique(combined_groups)
        for group in unique_groups:
            self.group_data_[group] = data[combined_groups == group]
            
        self.fitted_ = True
    
    def predict(self, alpha: float, grid_size: int = 1000) -> dict:
        """
        Predicts intervals for each group centered around the global mean.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        intervals = {}
        
        for group, data in self.group_data_.items():
            n = len(data)
            
            # Define grid for this group
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            buffer = data_range * 0.5 if data_range > 0 else 1.0
            y_grid = np.linspace(data_min - buffer, data_max + buffer, grid_size)
            
            # Calculate required rank for this group
            k = math.ceil((n + 1) * (1 - alpha))
            
            valid_ys = []
            for y in y_grid:
                # Use global mean instead of conditional mean
                residuals = np.abs(data - self.global_mean_)
                y_residual = abs(y - self.global_mean_)
                rank = np.sum(residuals < y_residual) + 1
                if rank <= k:
                    valid_ys.append(y)
            
            if valid_ys:
                intervals[group] = (min(valid_ys), max(valid_ys))
            else:
                raise RuntimeError(f"No valid intervals found for group {group}")
                
        return intervals

def plot_mondrian_intervals(y_test, groups_test, intervals, global_mean):
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis", 3)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Combine groups in test data
    groups_test_combined = groups_test.copy()
    groups_test_combined[groups_test <= 4] = 4
    groups_test_combined[groups_test >= 9] = 9
    
    # Sort everything by group for visualization
    sort_idx = np.argsort(groups_test_combined)
    y_test_sorted = y_test[sort_idx]
    groups_test_sorted = groups_test_combined[sort_idx]
    
    # Scatter plot
    ax.scatter(
        range(len(y_test)),
        y_test_sorted,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Plot intervals for each group
    current_idx = 0
    for group in sorted(intervals.keys()):
        group_mask = groups_test_sorted == group
        group_size = np.sum(group_mask)
        
        if group_size > 0:
            lower, upper = intervals[group]
            ax.fill_between(
                range(current_idx, current_idx + group_size),
                [lower] * group_size,
                [upper] * group_size,
                color=palette[1],
                alpha=0.4,
                label="Prediction Interval" if current_idx == 0 else ""
            )
            
            # Plot global mean
            ax.hlines(
                global_mean,
                current_idx,
                current_idx + group_size,
                colors='red',
                linestyles='--',
                label="Global Mean" if current_idx == 0 else ""
            )
            
            # Calculate coverage for annotation
            group_y = y_test[groups_test_combined == group]
            coverage = np.mean((group_y >= lower) & (group_y <= upper))
            interval_length = upper - lower
            
            # Add annotation for this group
            mid_point = current_idx + group_size/2
            
            # Determine label text based on group
            if group == 4:
                group_label = "Groups 1-4"
            elif group == 9:
                group_label = "Groups 9-10"
            else:
                group_label = f"Group {group}"
            
            # Determine if annotation should be above or below
            if group == 9:  # Place these groups' labels below
                y_pos = lower
                vert_offset = -10
                vert_align = 'top'
            else:  # Place other groups' labels above
                y_pos = upper
                vert_offset = 10
                vert_align = 'bottom'
            
            ax.annotate(
                f'{group_label}\nCoverage: {coverage:.1%}\nLength: {interval_length:.2f}',
                xy=(mid_point, y_pos),
                xytext=(0, vert_offset),
                textcoords='offset points',
                ha='center',
                va=vert_align,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                fontsize=8
            )
            
            current_idx += group_size
    
    # Labels and title
    ax.set_xlabel("Sample Index (sorted by Overall Quality)", fontsize=14)
    ax.set_ylabel("Log House Value ($)", fontsize=14)
    ax.set_title("Mondrian Conformal Prediction Intervals by Overall Quality", fontsize=18, weight='bold')
    
    # Add overall coverage annotation
    overall_coverage = np.mean([
        (y_test[groups_test_combined == group] >= intervals[group][0]) & 
        (y_test[groups_test_combined == group] <= intervals[group][1])
        for group in intervals.keys()
    ])
    
    ax.text(0.02, 0.98, f'Overall Coverage: {overall_coverage:.1%}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            verticalalignment='top')
    
    ax.legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plot_path + "mondrian_mean_prediction_interval.png", dpi=300)
    plt.show()

# Fit and predict
mcm = MondrianConformalMean()
overall_qual_train = X_train['OverallQual'].values
overall_qual_test = X_test['OverallQual'].values

mcm.fit(y_train, overall_qual_train)
intervals = mcm.predict(alpha=0.05)

# Calculate mean by group for visualization
mean_by_group = {group: np.mean(data) for group, data in mcm.group_data_.items()}

# Plot results
plot_mondrian_intervals(y_test, overall_qual_test, intervals, mcm.global_mean_)

# Calculate and print coverage by group
for group in sorted(intervals.keys()):
    group_mask = overall_qual_test == group
    group_y = y_test[group_mask]
    lower, upper = intervals[group]
    coverage = np.mean((group_y >= lower) & (group_y <= upper))
    interval_length = upper - lower
    print(f"Group {group}:")
    print(f"  Coverage: {coverage:.3f}")
    print(f"  Interval length: {interval_length:.3f}")
# %%
