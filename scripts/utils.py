import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

def load_prep_data(
    polynomial: bool = False,
    degree: int = 2
) -> pd.DataFrame:
    """
    This function loads the cleaned housing data from the CSV file located at "../data/cleaned/housing_clean.csv".
    
    Parameters
    ----------
    polynomial : bool
        Whether to generate polynomial features using the PolynomialFeatures.
    degree : int
        The degree of the polynomial features to generate.
    
    Returns
    -------
    pd.DataFrame
        The loaded and possibly transformed DataFrame.
    """
    path = "../data/cleaned/housing_clean.csv"

    df = pd.read_csv(path)
   

    if polynomial:
        X = df.drop('SalePrice', axis=1)  # features
        y = df['SalePrice']  # target variable
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(input_features=X.columns))
        
        df = pd.concat([X_poly, y], axis=1)  # combine transformed features with target variable

    return df

def prep_conditional_data(df: pd.DataFrame):
    """
    Prepare the conditional data for the conditional quantile regression model.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    
    Returns
    -------
    np.ndarray
        The features for the conditional quantile regression model.
    np.ndarray
        The target variable for the conditional quantile regression model.
    """
    def _combine_groups(groups):
        """
        Helper method to combine specified groups.
        """
        new_groups = groups.copy()
        # Combine groups 1-4
        new_groups[groups <= 4] = 4
        # Combine groups 9-10
        new_groups[groups >= 9] = 9
        return new_groups
    
    groups = df['OverallQual'].values
    y = np.log(df['SalePrice'].values)

    # Combine groups
    groups = _combine_groups(groups)

    # Turn into dummy variables using one-hot encoding
    # groups = pd.get_dummies(groups, drop_first=False).values
    X = OneHotEncoder().fit_transform(groups.reshape(-1, 1)).toarray()
    return X, y, groups


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


def sort_y_values(y_test, y_pred, y_pis):
    """
    Sorting the dataset in order to make plots using the fill_between function.
    """
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0, 0][indices]
    y_upper_bound = y_pis[:, 1, 0][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def train_cal_test_split(X, y, test_size=0.2, cal_size=0.2, random_state=42):
    """
    Split arrays or matrices into train, calibration, and test subsets.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    cal_size : float, default=0.2
        Proportion of the dataset to include in the calibration split
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    X_train : array-like
        Training data
    X_cal : array-like
        Calibration data
    X_test : array-like
        Test data
    y_train : array-like
        Training targets
    y_cal : array-like
        Calibration targets
    y_test : array-like
        Test targets
    """
    
    # Validate input parameters
    if not 0 <= test_size <= 1:
        raise ValueError("test_size should be between 0 and 1")
    if not 0 <= cal_size <= 1:
        raise ValueError("cal_size should be between 0 and 1")
    if test_size + cal_size >= 1:
        raise ValueError("Sum of test_size and cal_size should be less than 1")
    
    # Calculate the split points
    train_size = 1 - (test_size + cal_size)
    
    # If X is a pandas DataFrame or Series, get the index
    if hasattr(X, 'index'):
        indices = X.index
    else:
        indices = np.arange(len(X))
    
    # Shuffle the indices
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    
    # Calculate split points
    train_end = int(train_size * len(indices))
    cal_end = int((train_size + cal_size) * len(indices))
    
    # Split indices
    train_idx = shuffled_indices[:train_end]
    cal_idx = shuffled_indices[train_end:cal_end]
    test_idx = shuffled_indices[cal_end:]
    
    # Split X
    if hasattr(X, 'iloc'):  # If X is a pandas DataFrame
        X_train = X.iloc[train_idx]
        X_cal = X.iloc[cal_idx]
        X_test = X.iloc[test_idx]
    else:  # If X is a numpy array
        X_train = X[train_idx]
        X_cal = X[cal_idx]
        X_test = X[test_idx]
    
    # Split y
    if hasattr(y, 'iloc'):  # If y is a pandas Series
        y_train = y.iloc[train_idx]
        y_cal = y.iloc[cal_idx]
        y_test = y.iloc[test_idx]
    else:  # If y is a numpy array
        y_train = y[train_idx]
        y_cal = y[cal_idx]
        y_test = y[test_idx]
    
    return X_train, X_cal, X_test, y_train, y_cal, y_test


class QuantileRegressorCV(QuantileRegressor):
    def __init__(self, alphas=None, cv=5, quantile=0.5, **kwargs):
        super().__init__(quantile=quantile, **kwargs)
        if alphas is None:
            alphas = np.logspace(-3, 1, 10)
        self.alphas = alphas
        self.cv = cv
        self.best_alpha = None
        self.best_model = None

    def fit(self, X, y):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        avg_loss = []

        for alpha in self.alphas:
            loss_list = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = QuantileRegressor(alpha=alpha, quantile=self.quantile, solver='highs')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                loss_list.append(mean_pinball_loss(y_val, y_pred, alpha=self.quantile))

            avg_loss.append(np.mean(loss_list))

        self.best_alpha = self.alphas[np.argmin(avg_loss)]
        self.alpha = self.best_alpha
        QuantileRegressor.fit(self, X, y)
        self.best_model = self

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with appropriate data.")
        return super().predict(X)

    def get_best_alpha(self):
        if self.best_alpha is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' with appropriate data.")
        return self.best_alpha


# # Example usage:
# from sklearn.datasets import make_regression
# X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=42)
# model = QuantileRegressorCV(quantile = 0.90)
# model.fit(X, y)
# predictions = model.predict(X)
# best_alpha = model.get_best_alpha()

# coefs_ = model.coef_
# # round to zero if less than 0.001
# coefs_ = np.where(np.abs(coefs_) < 0.001, 0, coefs_)
# print("Number of non-zero coefficients:", np.count_nonzero(model.coef_))

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
    
    # Plot CIs that contain true value (gray) with green dots
    ax.errorbar(y_test[~outside_ci], y_pred[~outside_ci], 
               yerr=[ci_lower[~outside_ci], ci_upper[~outside_ci]], 
               fmt='o', ecolor='gray', alpha=0.5, label='CI (True Value Inside)', color='green', mfc='green', mec='green', mew=0.5, ms=5)
    
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
# Example usage with subplots:
"""
# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot first method
plot_predictions_with_ci(y_test_method1, y_pred_method1, 
                        ci_lower_method1, ci_upper_method1, 
                        ax=ax1, title="Method 1")

# Plot second method
plot_predictions_with_ci(y_test_method2, y_pred_method2, 
                        ci_lower_method2, ci_upper_method2, 
                        ax=ax2, title="Method 2")

plt.tight_layout()
plt.show()
"""

def plot_conditional_intervals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pis: np.ndarray,
    groups_test: np.ndarray,
    title: str = "Group-Specific Predictions with Conformal Prediction Intervals"
) -> None:
    """
    Plot prediction intervals with group-specific coverage and length metrics.

    This function creates a visualization of prediction intervals for different groups,
    showing actual values, predictions, and confidence intervals. It includes
    annotations for group-specific coverage rates and interval lengths.

    Parameters
    ----------
    y_test : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted mean values
    y_pis : np.ndarray
        Prediction intervals array of shape (n_samples, 2, n_intervals)
        where [:, 0, :] are lower bounds and [:, 1, :] are upper bounds
    groups_test : np.ndarray
        Array of group labels for each test sample

    Returns
    -------
    None
        Displays the plot using matplotlib
    """
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
    y_pred_sorted = y_pred[sort_idx]
    y_pis_sorted = y_pis[sort_idx]
    groups_test_sorted = groups_test_combined[sort_idx]
    
    # Scatter plot of actual values
    ax.scatter(
        range(len(y_test)),
        y_test_sorted,
        label="Test Data",
        color=palette[0],
        alpha=0.8,
        edgecolor="k",
        s=60
    )
    
    # Calculate overall coverage and average length
    in_interval = (y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0])
    overall_coverage = np.mean(in_interval)
    average_length = np.mean(y_pis[:, 1, 0] - y_pis[:, 0, 0])
    
    # Plot continuous prediction intervals and mean
    x_range = np.arange(len(y_test))
    
    # Plot prediction intervals as continuous lines
    ax.fill_between(
        x_range,
        y_pis_sorted[:, 0, 0],
        y_pis_sorted[:, 1, 0],
        color=palette[1],
        alpha=0.4,
        label="Prediction Interval"
    )
    
    # Plot mean prediction as continuous line
    ax.plot(
        x_range,
        y_pred_sorted,
        color='red',
        linestyle='--',
        label="Conditional Prediction",
        linewidth=2
    )
    
    # Add group annotations
    current_idx = 0
    for group in sorted(np.unique(groups_test_combined)):
        group_mask = groups_test_sorted == group
        group_size = np.sum(group_mask)
        
        if group_size > 0:
            # Calculate group-specific metrics
            group_y = y_test_sorted[group_mask]
            group_lower = y_pis_sorted[group_mask, 0, 0]
            group_upper = y_pis_sorted[group_mask, 1, 0]
            group_coverage = np.mean((group_y >= group_lower) & (group_y <= group_upper))
            interval_length = np.mean(group_upper - group_lower)
            
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
                y_pos = np.min(group_lower)
                vert_offset = -10
                vert_align = 'top'
            else:  # Place other groups' labels above
                y_pos = np.max(group_upper)
                vert_offset = 10
                vert_align = 'bottom'
            
            ax.annotate(
                f'{group_label}\nCoverage: {group_coverage:.1%}\nLength: {interval_length:.2f}',
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
    ax.set_title(title, fontsize=18, weight='bold')
    
    # Add overall coverage and average length annotations
    ax.text(0.02, 0.98, 
            f'Overall Coverage: {overall_coverage:.1%}\nAverage Length: {average_length:.2f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            verticalalignment='top')
    
    ax.legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.show()