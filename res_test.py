# %%
from Residual_Cascade_Boosting import ResidualCascadeBoosting
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
# %%

# Create a synthetic dataset
X, y = make_friedman2(n_samples=1000, random_state=42)




# Create the ResidualCascadeBoosting model
rcb = ResidualCascadeBoosting(
    base_estimators=[LinearRegression(), Lasso(), RandomForestRegressor()],
    n_iterations=10,
    learning_rate=0.1,
    random_state=42
)

# %%
rcb.fit(X, y)

# %%
# Use cross-validation to evaluate the model
scores = cross_val_score(rcb, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
# %%
# Compare to just a single RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
# %%
# Compare to ensemble using the same base estimators
from sklearn.ensemble import StackingRegressor

stacking = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('lasso', Lasso()),
        ('rf', RandomForestRegressor())
    ],
    final_estimator=LinearRegression()
)

scores = cross_val_score(stacking, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
# %%
