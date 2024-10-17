from typing import List, Union, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone

class ResidualCascadeBoosting(BaseEstimator, RegressorMixin):
    """Residual Cascade Boosting Regressor.

    This class implements the Residual Cascade Boosting algorithm, which combines
    elements of both stacking and boosting. It iteratively fits a sequence of
    models on the residuals of the previous model's predictions.

    Parameters
    ----------
    base_estimators : List[BaseEstimator]
        List of scikit-learn compatible estimators to be used in the ensemble.
    n_iterations : int, default=10
        The number of boosting iterations.
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each regressor.
    random_state : Optional[int], default=None
        Controls the randomness of the bootstrapping of the samples used
        when building trees.

    Attributes
    ----------
    estimators_ : List[BaseEstimator]
        The collection of fitted sub-estimators.
    train_score_ : ndarray, shape (n_iterations,)
        The i-th score is the training score for the model at the i-th iteration.
    val_score_ : ndarray, shape (n_iterations,)
        The i-th score is the validation score for the model at the i-th iteration.
    """

    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        n_iterations: int = 10,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.base_estimators = base_estimators
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build a Residual Cascade Boosting model from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Split the data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        self.estimators_ = []
        self.train_score_ = np.zeros(self.n_iterations)
        self.val_score_ = np.zeros(self.n_iterations)

        residuals = y_train.copy()
        val_residuals = y_val.copy()
        train_predictions = np.zeros_like(y_train)
        val_predictions = np.zeros_like(y_val)

        for i in range(self.n_iterations):
            model = clone(self.base_estimators[i % len(self.base_estimators)])
            model.fit(X_train, residuals)
            self.estimators_.append(model)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_predictions += self.learning_rate * train_pred
            val_predictions += self.learning_rate * val_pred

            residuals -= self.learning_rate * train_pred
            val_residuals -= self.learning_rate * val_pred

            self.train_score_[i] = mean_squared_error(y_train, train_predictions)
            self.val_score_[i] = mean_squared_error(y_val, val_predictions)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but this model "
                             f"was trained with {self.n_features_in_} features.")

        predictions = np.zeros(X.shape[0])
        for model in self.estimators_:
            predictions += self.learning_rate * model.predict(X)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return super().score(X, y)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return super().get_params(deep)

    def set_params(self, **params) -> 'ResidualCascadeBoosting':
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        return super().set_params(**params)
