
# %%
import numpy as np
import pandas as pd
# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple

# Data Generating Process (DGP)
def generate_data(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(n_samples) * 0.5
    return X, y

# Utility Function
def utility_function(y_true: float, y_pred: float) -> float:
    return -abs(y_true - y_pred)  # Negative absolute error

# Conformal Predictive Distribution (CPD) Function
def compute_cpd(
    training_sequence: List[Tuple[np.ndarray, float]],
    test_object: np.ndarray,
    alpha: float = 0.1
) -> Tuple[float, float]:
    X = np.array([x for x, _ in training_sequence])
    y = np.array([y for _, y in training_sequence])

    # Split the data into training and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Compute nonconformity scores on the calibration set
    y_pred_calib = model.predict(X_calib)
    nonconformity_scores = np.abs(y_calib - y_pred_calib)

    # Compute prediction interval for the test object
    y_pred_test = model.predict(test_object.reshape(1, -1))[0]
    quantile = np.quantile(nonconformity_scores, 1 - alpha)

    lower_bound = y_pred_test - quantile
    upper_bound = y_pred_test + quantile

    return lower_bound, upper_bound

# Conformal Predictive Decision Making Algorithm
def conformal_predictive_decision_making(
    training_sequence: List[Tuple[np.ndarray, float]],
    test_object: np.ndarray,
    decision_space: List[float],
    utility_function: callable,
    cpd_function: callable
) -> float:
    best_decision = None
    best_utility = float('-inf')

    for d in decision_space:
        lower_bound, upper_bound = cpd_function(training_sequence, test_object)

        # Compute expected utility over the prediction interval
        expected_utility = np.mean([utility_function(y, d) for y in np.linspace(lower_bound, upper_bound, 100)])

        if expected_utility > best_utility:
            best_utility = expected_utility
            best_decision = d

    return best_decision




# %%
#
# # Example data
# Simulation
np.random.seed(42)

# Generate data
n_samples, n_features = 1000, 5
X, y = generate_data(n_samples, n_features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Prepare training sequence
training_sequence = list(zip(X_train, y_train))

# Define decision space
decision_space = np.linspace(y.min(), y.max(), 100)

# Run simulation
results = []
for test_object, true_y in zip(X_test, y_test):
    best_decision = conformal_predictive_decision_making(
        training_sequence,
        test_object,
        decision_space,
        utility_function,
        compute_cpd
    )
    actual_utility = utility_function(true_y, best_decision)
    results.append((true_y, best_decision, actual_utility))

# Analyze results
true_values, predictions, utilities = zip(*results)
mse = np.mean((np.array(true_values) - np.array(predictions)) ** 2)
mae = np.mean(np.abs(np.array(true_values) - np.array(predictions)))
mean_utility = np.mean(utilities)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Utility: {mean_utility:.4f}")

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Conformal Predictive Decision Making: True vs Predicted")
plt.show()



# %%
car_loan = 96
c_interest = 0.09

student_loan = 20
s_interest = 0.09

# %%
# Optimal payment strategy
# Initial loan amounts and interest rates
car_loan = 96_000_000
c_interest = 0.09
car_weeks = 210

student_loan = 20_000_000
s_interest = 0.09
student_weeks = 62

extra_payment = 20_000_000


# Function to calculate total interest paid
def calculate_total_interest(loan_amount, interest_rate, weeks, extra_payment=0):
    total_interest = 0
    loan_amount -= extra_payment
    for _ in range(weeks):
        interest = loan_amount * (interest_rate / 52)  # Weekly interest
        total_interest += interest
        payment = loan_amount / weeks
        loan_amount -= payment
        if loan_amount <= 0:
            break
    return total_interest

# Scenario 1: Extra payment to car loan
car_interest_1 = calculate_total_interest(car_loan, c_interest, car_weeks, extra_payment)
student_interest_1 = calculate_total_interest(student_loan, s_interest, student_weeks)
total_interest_1 = car_interest_1 + student_interest_1

# Scenario 2: Extra payment to student loan
car_interest_2 = calculate_total_interest(car_loan, c_interest, car_weeks)
student_interest_2 = calculate_total_interest(student_loan, s_interest, student_weeks, extra_payment)
total_interest_2 = car_interest_2 + student_interest_2

# Compare and print results
if total_interest_1 < total_interest_2:
    print(f"You should pay the extra ${extra_payment:.2f} into the car loan.")
    print(f"Total interest paid: ${total_interest_1:.2f}")
else:
    print(f"You should pay the extra ${extra_payment:.2f} into the student loan.")
    print(f"Total interest paid: ${total_interest_2:.2f}")

print(f"\nInterest if paying extra to car loan: ${total_interest_1:.2f}")
print(f"Interest if paying extra to student loan: ${total_interest_2:.2f}")
