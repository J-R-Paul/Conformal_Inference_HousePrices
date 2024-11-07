# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)  # For reproducibility
n_simulations = 1000  # Number of Monte Carlo simulations
sample_size = 5  # Sample size
alpha = 0.30  # Significance level (for 95% confidence intervals)
true_mean = 0  # True mean of the distribution
true_std = 1  # True standard deviation

# Grid parameters for full conformal
n_grid = 1_0  # Number of grid points
grid_width = 4  # Number of standard deviations to extend grid

# Storage for results
coverage_conformal = 0
coverage_conformal_ROO = 0
interval_widths_conformal = []
interval_widths_ROOP = []

# Define quantile functions
def max_quantile(vs, alpha=0.05):
    """Compute the (1 - alpha)-quantile for conformity scores."""
    vs_ordered = np.sort(vs)
    return vs_ordered[int(np.ceil((1 - alpha) * len(vs))) - 1]




for sim in range(n_simulations):
    # Generate a random sample from a normal distribution
    y = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    
    # Estimate the mean
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    # --- Full Conformal Method ---
    # Create a dense grid of possible y values
    y_grid = np.linspace(
        y_mean - grid_width * y_std,
        y_mean + grid_width * y_std,
        n_grid
    )
    # y_grid = y
    # Function to compute conformity scores
    def compute_conformity_scores(y_aug):
        """Compute conformity scores for augmented dataset."""
        mean_aug = np.mean(y_aug)
        return np.abs(y_aug - mean_aug)
    
    valid_points = []
    valid_points_roo = []
    print('-'*5)
    for y_trial in y_grid:
        print("y_trial: ", y_trial)
        # Augment the dataset with the trial value
        y_aug = np.append(y, y_trial)
        
        # Compute conformity scores
        scores = compute_conformity_scores(y_aug)
        
        # ---------------- Conformal Prediction ----------------
        # Calculate the number of scores less than or equal to the trial score
        lhs = np.sum(scores <= scores[-1])
        print(lhs)
        
        # Calculate the threshold
        rhs = np.ceil((sample_size + 1) * (1 - alpha))
        print(rhs)
        # Determine if the trial value is included in the prediction set
        include_trial = lhs <= rhs
        print(include_trial)
        
        if include_trial:
            valid_points.append(y_trial)
        
        # ------------------- Conformal ROO (ROOP) -------------------
        # Compute the (1 - alpha)-quantile of the conformity scores
        q = max_quantile(scores, alpha=alpha)
        
        # Compute the augmented mean
        mean_aug = np.mean(y_aug)
        
        # Define the prediction interval based on ROOP
        L_lower_j = mean_aug - q
        L_upper_j = mean_aug + q
        
        # Check if the trial value falls within the ROOP interval
        if L_lower_j <= y_trial <= L_upper_j:
            valid_points_roo.append(y_trial)
    
    test_sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    # Construct the conformal confidence interval
    if valid_points:
        CI_conformal = [np.min(valid_points), np.max(valid_points)]
        interval_widths_conformal.append(CI_conformal[1] - CI_conformal[0])
        
        # Check coverage on a hold-out sample
        coverage_conformal += np.mean((test_sample >= CI_conformal[0]) & 
                                       (test_sample <= CI_conformal[1]))
    else:
        # Handle cases with no valid points
        interval_widths_conformal.append(np.nan)
    
    # Construct the ROOP confidence interval
    if valid_points_roo:
        CI_conformal_ROO = [np.min(valid_points_roo), np.max(valid_points_roo)]
        interval_widths_ROOP.append(CI_conformal_ROO[1] - CI_conformal_ROO[0])
        
        # Check coverage on a hold-out sample
        coverage_conformal_ROO += np.mean((test_sample >= CI_conformal_ROO[0]) & 
                                          (test_sample <= CI_conformal_ROO[1]))
    else:
        # Handle cases with no valid points
        interval_widths_ROOP.append(np.nan)

    if CI_conformal[0] != CI_conformal_ROO[0]:
        print(f"Break on simulation {sim}: CI_conformal[0] != CI_conformal_ROO[0]")
        print(f"Upper bound: {CI_conformal_ROO[1], CI_conformal[1]}")
        print(f"Lower bound: {CI_conformal_ROO[0], CI_conformal[0]}")
        # Check to see if there are ties in scores
        print(f"y trial: {y_trial}")
        scores = compute_conformity_scores(y)
        unique_scores, counts = np.unique(scores, return_counts=True)
        if np.any(counts > 1):
            print(f"Ties found in scores for simulation {sim}.")
        
        # Check if any y_trial is exactly equal to the bounds of the ROOP interval
        lower_bound_match = y_grid[np.isclose(y_grid, CI_conformal_ROO[0], atol=1e-6)]
        upper_bound_match = y_grid[np.isclose(y_grid, CI_conformal_ROO[1], atol=1e-6)]
        
        if lower_bound_match.size > 0:
            print(f"y_trial {lower_bound_match[0]} is equal to the lower bound of the ROOP interval for simulation {sim}.")
        if upper_bound_match.size > 0:
            print(f"y_trial {upper_bound_match[0]} is equal to the upper bound of the ROOP interval for simulation {sim}.")
        break
    elif CI_conformal[1] != CI_conformal_ROO[1]:
        print(f"Break on simulation {sim}: CI_conformal[1] != CI_conformal_ROO[1]")
        print(f"Upper bound: {CI_conformal_ROO[1], CI_conformal[1]}")
        print(f"Lower bound: {CI_conformal_ROO[0], CI_conformal[0]}")
        print(f"y trial: {y_trial}")
        scores = compute_conformity_scores(y)
        unique_scores, counts = np.unique(scores, return_counts=True)
        if np.any(counts > 1):
            print(f"Ties found in scores for simulation {sim}.")


        # Check if any y_trial is exactly equal to the bounds of the ROOP interval
        lower_bound_match = y_grid[np.isclose(y_grid, CI_conformal_ROO[0], atol=1e-6)]
        upper_bound_match = y_grid[np.isclose(y_grid, CI_conformal_ROO[1], atol=1e-6)]
        
        if lower_bound_match.size > 0:
            print(f"y_trial {lower_bound_match[0]} is equal to the lower bound of the ROOP interval for simulation {sim}.")
        if upper_bound_match.size > 0:
            print(f"y_trial {upper_bound_match[0]} is equal to the upper bound of the ROOP interval for simulation {sim}.")
    

    # Check if y_trial is within epsilon of the bounds of the ROOP interval
        if abs(CI_conformal_ROO[0] - y_trial) < 1e-6 or abs(y_trial - CI_conformal_ROO[1]) < 1e-6:
            print(f"y_trial is within epsilon of the bound of the ROOP interval for simulation {sim}.")

        break


        



    




# Calculate coverage probabilities
coverage_prob_conformal = coverage_conformal / n_simulations
coverage_prob_ROOP = coverage_conformal_ROO / n_simulations

# Calculate average interval widths
avg_interval_width_conformal = np.nanmean(interval_widths_conformal)
avg_interval_width_ROOP = np.nanmean(interval_widths_ROOP)

# Print the results
print(f"Results after {n_simulations} simulations:")

print("\nFull Conformal Method:")
print(f"Coverage Probability: {coverage_prob_conformal:.3f}")
print(f"Average Interval Width: {avg_interval_width_conformal:.3f}")

print("\nConformal ROO (ROOP) Method:")
print(f"Coverage Probability: {coverage_prob_ROOP:.3f}")
print(f"Average Interval Width: {avg_interval_width_ROOP:.3f}")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Distribution of interval widths
ax1.hist(interval_widths_conformal, bins=30, alpha=0.6, label='Conformal', edgecolor='green')
ax1.hist(interval_widths_ROOP, bins=30, alpha=0.6, label='Conformal ROO (ROOP)', edgecolor='red')
ax1.set_xlabel('Interval Width')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Confidence Interval Widths')
ax1.legend()

# Plot 2: Coverage comparison
methods = ['Conformal', 'Conformal ROO (ROOP)']
coverages = [coverage_prob_conformal, coverage_prob_ROOP]
widths = [avg_interval_width_conformal, avg_interval_width_ROOP]

x = np.arange(len(methods))
width_bar = 0.35

# Create bar plots for coverage and average width
ax2.bar(x - width_bar/2, coverages, width_bar, label='Coverage Probability', color='skyblue')
ax2.bar(x + width_bar/2, widths, width_bar, label='Average Interval Width', color='salmon')
ax2.axhline(y=1 - alpha, color='gray', linestyle='--', label='Target Coverage')
ax2.set_ylabel('Probability / Width')
ax2.set_title('Coverage Probability and Average Interval Width')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()

plt.tight_layout()
plt.show()

# %%
# Y_aug [ 0.35778736  0.56078453  1.08305124  1.05380205 -1.37766937  3.94043222]
y = np.array([ 0.35778736,  0.56078453,  1.08305124,  1.05380205, -1.37766937])
trial_vals = np.array([-3.26932989, -2.46824521, -1.66716053, -0.86607585, -0.06499118,
        0.7360935 ,  1.53717818,  2.33826286,  3.13934754,  3.94043222])

trial = trial_vals[6]

# Break on simulation 20: CI_conformal[0] != CI_conformal_ROO[0]
# Upper bound: (np.float64(2.3382628586927403), np.float64(2.3382628586927403))
# Lower bound: (np.float64(-0.06499117638584284), np.float64(-0.8660758547453709))
# %
y_aug = np.append(y, trial)
# Compute conformity scores
scores = compute_conformity_scores(y_aug)
print(scores)

lhs = np.sum(scores <= scores[-1])
print(f"LHS {lhs}")
rhs = np.ceil((len(y)+1) * (1-alpha))
print(f"rhs {rhs}")

print(f"Include: {lhs <= rhs}")


# print(f'critical value {y_aug[rhs]}')
np.sort(scores)[int(rhs)-1]
# [int(rhs)-1]

print("----"*4)

# %
q = max_quantile(scores, alpha=alpha)
print("Estimated Quantile: ", q)

# Compute the augmented mean
mean_aug = np.mean(y_aug)

# Define the prediction interval based on ROOP
L_lower_j = mean_aug - q
print(f'lower: {L_lower_j}')
L_upper_j = mean_aug + q
print(f"upper: {L_upper_j}")
print(f"trial: {trial}")

print(f"include: {L_lower_j <= trial <= L_upper_j}")

# %%
trial


# %%
import numpy as np

def vovk_include(nonconf_scores, trial_score, alpha):
    n = len(nonconf_scores)
    p_value = np.sum(nonconf_scores >= trial_score) / (n + 1)
    return p_value > alpha

def rank_include(nonconf_scores, trial_score, alpha):
    n = len(nonconf_scores)
    rank = np.sum(nonconf_scores <= trial_score)
    threshold = np.ceil((1-alpha)*(n+1))
    return rank <= threshold

# Simple Monte Carlo
np.random.seed(42)
n = 100  # calibration set size
alpha = 0.05  # significance level

# Generate some random nonconformity scores
n_trials = 1000
differences = 0

for _ in range(n_trials):
    nonconf_scores = np.random.randn(n)
    trial_score = np.random.randn()
    
    vovk_decision = vovk_include(nonconf_scores, trial_score, alpha)
    rank_decision = rank_include(nonconf_scores, trial_score, alpha)
    
    if vovk_decision != rank_decision:
        differences += 1
        
        # Let's examine a few cases where they differ
        if differences <= 3:  # Just look at first 3 differences
            print("\nFound difference:")
            print(f"Trial score: {trial_score}")
            print(f"Vovk p-value: {np.sum(nonconf_scores >= trial_score)/(n+1)}")
            print(f"Rank: {np.sum(nonconf_scores <= trial_score)}")
            print(f"Threshold for rank: {np.ceil((1-alpha)*(n+1))}")
            print(f"Vovk decision: {vovk_decision}")
            print(f"Rank decision: {rank_decision}")

print(f"\nTotal differences: {differences} out of {n_trials} trials ({100*differences/n_trials:.2f}%)")
# %%
import numpy as np

def vovk_include(nonconf_scores, trial_score, alpha):
    n = len(nonconf_scores)
    p_value = np.sum(nonconf_scores >= trial_score) / (n + 1)
    return p_value > alpha

def rank_include(nonconf_scores, trial_score, alpha):
    n = len(nonconf_scores)
    rank = np.sum(nonconf_scores <= trial_score)
    threshold = np.ceil((1-alpha)*(n+1))
    return rank <= threshold

# Monte Carlo to check coverage and interval width
np.random.seed(42)
n = 100  # calibration set size
alpha = 0.1  # significance level
n_trials = 1000

# We'll generate data from standard normal for simplicity
vovk_coverage = 0
rank_coverage = 0
vovk_widths = []
rank_widths = []

for _ in range(n_trials):
    # Generate calibration data
    cal_scores = np.random.randn(n)
    # True value we want to cover
    true_value = np.random.randn()
    
    # Check coverage by testing a grid of points
    grid = np.linspace(true_value-5, true_value+5, 200)
    
    vovk_set = [y for y in grid if vovk_include(cal_scores, y, alpha)]
    rank_set = [y for y in grid if rank_include(cal_scores, y, alpha)]
    
    # Check coverage
    vovk_coverage += min(vovk_set) <= true_value <= max(vovk_set)
    rank_coverage += min(rank_set) <= true_value <= max(rank_set)
    
    # Record widths
    vovk_widths.append(max(vovk_set) - min(vovk_set))
    rank_widths.append(max(rank_set) - min(rank_set))

print(f"Vovk method coverage: {vovk_coverage/n_trials:.3f}")
print(f"Rank method coverage: {rank_coverage/n_trials:.3f}")
print(f"Vovk method average width: {np.mean(vovk_widths):.3f}")
print(f"Rank method average width: {np.mean(rank_widths):.3f}")
# %%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Parameters
n = 100  # number of points to visualize
threshold = 5  # our (1-alpha)(n+1) threshold
trials = 1000  # number of random trials for each point

# Generate some C values around the threshold
C_values = np.linspace(threshold-1, threshold+1, n)

# Sharp cutoff (p-trial method)
sharp_inclusion = C_values < threshold

# Smoothed version
smoothed_inclusions = np.zeros(n)
for i, C in enumerate(C_values):
    taus = np.random.uniform(0, 1, trials)
    smoothed_inclusions[i] = np.mean(C + taus < threshold)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(C_values, sharp_inclusion, 'b-', label='Sharp cutoff', alpha=0.5)
plt.plot(C_values, smoothed_inclusions, 'r-', label='Smoothed (randomized)')
plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
plt.xlabel('C value')
plt.ylabel('Probability of inclusion')
plt.title(f'Sharp vs Smoothed Cutoff at threshold = {threshold}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
