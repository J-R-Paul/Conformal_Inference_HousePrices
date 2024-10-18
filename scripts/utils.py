import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


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
