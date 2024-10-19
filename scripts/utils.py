import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss



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



# %% 


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

# %%
