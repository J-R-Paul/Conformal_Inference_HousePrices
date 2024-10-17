# %%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_prep_data

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor

# %% 
df = load_prep_data(polynomial=False)

df.head()
df.columns
# %%

# X = df.drop("SalePrice", axis=1)
X = df["BedroomAbvGr"].to_numpy().reshape(-1, 1)
y = np.log(df["SalePrice"].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=123,
)

X_test, X_cal, y_test, y_cal = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=123,
)

# %%
lr = LinearRegression()

lr.fit(X_train, y_train)

mapie = MapieRegressor(lr, cv="split")
mapie.fit(X_cal, y_cal)
preds, pis = mapie.predict(X_test, alpha=0.05)

# %%




def plot_mapie(preds, pis, X, y):
    # Sort X and corresponding predictions and intervals
    sorted_indices = np.argsort(X.ravel())
    X_sorted = X.ravel()[sorted_indices]
    preds_sorted = preds[sorted_indices]
    lower_sorted = pis[:, 0].ravel()[sorted_indices]
    upper_sorted = pis[:, 1].ravel()[sorted_indices]
    y_sorted = y[sorted_indices]

    # Plot test data, predictions, and prediction intervals
    fig, ax = plt.subplots()
    ax.scatter(X_sorted, y_sorted, label="test data")
    ax.scatter(X_sorted, preds_sorted, label="predictions", color="orange")
    ax.fill_between(
        X_sorted,
        lower_sorted,
        upper_sorted,
        alpha=0.5,
        label="prediction interval",
        color="green"
    )
    ax.legend() 
    plt.show()

# Call the updated function
plot_mapie(preds, pis, X_test, y_test)


# %%
