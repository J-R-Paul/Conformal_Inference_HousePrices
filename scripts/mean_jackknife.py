# %%
# import pandas as pd 
# import numpy as np

# # %%
# # ys = np.array([1,1,2,1, 10])
# n = 100
# ys = np.random.normal(size=n) + 100*np.random.binomial(n=1, p=0.1, size=n)
# means = np.zeros(n)
# sample_mean = np.sum(ys)/n
# Rloos = np.zeros(n)

# for i in range(n):
#     # Mean excluding the i-th value
#     ys_not_i = ys[np.arange(n) != i]
#     means[i] = np.sum(ys_not_i)/(n-1)
#     # jackknife
#     Rloos[i] = np.abs(means[i] - ys[i])


# # %%
# means


# # %%
# Rloos


# # %%
# jKp_lb = np.zeros(n)
# jK_lb = np.zeros(n)
# jkp_ub = np.zeros(n)
# jk_ub = np.zeros(n)
# for i in range(n):
#     jKp_lb[i] = means[i] - Rloos[i]
#     jK_lb[i] = sample_mean - Rloos[i]

#     jkp_ub[i] = means[i] + Rloos[i]
#     jk_ub[i] = sample_mean + Rloos[i]



# # %%
# alpha = 0.05
# jkp_uq = np.quantile(jkp_ub, 1-alpha)
# jkp_lq = np.quantile(jKp_lb, alpha)
# print(f"JackKnife +: Lower bound: {jkp_lq}. Upper bound: {jkp_uq}")
# print(f"Lenght: {jkp_uq - jkp_lq}")
# # np.quantile(jK_lb, alpha)
# # print(np.sort(jKp_lb))
# # print(np.sort(jkp_ub))

# jk_uq = np.quantile(jk_ub, 1-alpha)
# jk_lq = np.quantile(jK_lb, alpha)
# print(f"JackKnife -: Lower bound: {jk_lq}. Upper bound: {jk_uq}")
# print(f"Lenght: {jk_uq - jk_lq}" )


# # print(np.sort(jK_lb))
# # print(np.sort(jk_ub))
# # %%
# 0.1 + 0.2
# # %%

# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Simulation parameters
# np.random.seed(0)            # For reproducibility
# n_simulations = 1000         # Number of Monte Carlo simulations
# sample_size = 300             # Sample size
# alpha = 0.05                 # Significance level (for 95% confidence intervals)
# true_mean = 0                # True mean of the distribution
# true_std = 1                # True standard deviation

# # Storage for results
# coverage_LOO = 0
# coverage_ROO = 0
# interval_widths_LOO = []
# interval_widths_ROO = []

# for sim in range(n_simulations):
#     # Generate a random sample from a normal distribution
#     y = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    
#     # True parameter to estimate
#     mu = true_mean
    
#     # Estimate the mean
#     y_mean = np.mean(y)
    
#     # --- LOO Method ---
#     residuals_LOO = []
#     for i in range(sample_size):
#         # Leave-one-out mean
#         y_mean_LOO = (np.sum(y) - y[i]) / (sample_size - 1)
#         # Residual
#         residual_LOO = np.abs(y[i] - y_mean_LOO)
#         residuals_LOO.append(residual_LOO)
    
#     # Compute the (1 - alpha)-quantile of the residuals
#     q_LOO = np.quantile(residuals_LOO, 1 - alpha)
#     # Construct the confidence interval
#     CI_LOO = [y_mean - q_LOO, y_mean + q_LOO]
#     interval_widths_LOO.append(CI_LOO[1] - CI_LOO[0])
#     # Check covarege on a hold-out sample
#     sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#     coverage_LOO += np.sum((sample >= CI_LOO[0]) & (sample <= CI_LOO[1])) / sample_size

    
#     # --- ROO Method ---
#     residuals_ROO = []
#     for i in range(sample_size):
#         # Repeat-one-observation mean
#         y_mean_ROO = (np.sum(y) + y[i]) / (sample_size + 1)
#         # Residual
#         residual_ROO = np.abs(y[i] - y_mean_ROO)
#         residuals_ROO.append(residual_ROO)
    
#     # Compute the (1 - alpha)-quantile of the residuals
#     q_ROO = np.quantile(residuals_ROO, 1 - alpha)
#     # Construct the confidence interval
#     CI_ROO = [y_mean - q_ROO, y_mean + q_ROO]
#     interval_widths_ROO.append(CI_ROO[1] - CI_ROO[0])
#     # Check covarege on a hold-out sample
#     sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#     coverage_ROO += np.sum((sample >= CI_ROO[0]) & (sample <= CI_ROO[1])) / sample_size

# # Calculate coverage probabilities
# coverage_prob_LOO = coverage_LOO / n_simulations
# coverage_prob_ROO = coverage_ROO / n_simulations

# # Calculate average interval widths
# avg_interval_width_LOO = np.mean(interval_widths_LOO)
# avg_interval_width_ROO = np.mean(interval_widths_ROO)

# # Print the results
# print("Results after {} simulations:".format(n_simulations))
# print("\nLOO Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_LOO))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_LOO))

# print("\nROO Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_ROO))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_ROO))

# # Plotting the distribution of interval widths
# plt.hist(interval_widths_LOO, bins=30, alpha=0.5, label='LOO Interval Widths')
# plt.hist(interval_widths_ROO, bins=30, alpha=0.5, label='ROO Interval Widths')
# plt.xlabel('Interval Width')
# plt.ylabel('Frequency')
# plt.title('Distribution of Confidence Interval Widths')
# plt.legend()
# plt.show()
# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # Simulation parameters
# np.random.seed(0)  # For reproducibility
# n_simulations = 1000  # Number of Monte Carlo simulations
# sample_size = 30  # Sample size
# alpha = 0.05  # Significance level (for 95% confidence intervals)
# true_mean = 0  # True mean of the distribution
# true_std = 1  # True standard deviation

# # Grid parameters for full conformal
# n_grid = 1000  # Number of grid points
# grid_width = 4  # Number of standard deviations to extend grid

# # Storage for results
# coverage_LOO = 0
# coverage_ROO = 0
# coverage_conformal = 0
# coverage_conformal_ROO = 0
# interval_widths_LOO = []
# interval_widths_ROO = []
# interval_widths_conformal = []
# interval_widths_ROOP = []

# for sim in range(1000):
#     # Generate a random sample from a normal distribution
#     y = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    
#     # True parameter to estimate
#     mu = true_mean
    
#     # Estimate the mean
#     y_mean = np.mean(y)
#     y_std = np.std(y)
    
#     # --- LOO Method ---
#     residuals_LOO = []
#     for i in range(sample_size):
#         # Leave-one-out mean
#         y_mean_LOO = (np.sum(y) - y[i]) / (sample_size - 1)
#         # Residual
#         residual_LOO = np.abs(y[i] - y_mean_LOO)
#         residuals_LOO.append(residual_LOO)
    
#     # Compute the (1 - alpha)-quantile of the residuals
#     q_LOO = np.quantile(residuals_LOO, 1 - alpha)
    
#     # Construct the confidence interval
#     CI_LOO = [y_mean - q_LOO, y_mean + q_LOO]
#     interval_widths_LOO.append(CI_LOO[1] - CI_LOO[0])
    
#     # Check coverage on a hold-out sample
#     sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#     coverage_LOO += np.sum((sample >= CI_LOO[0]) & (sample <= CI_LOO[1])) / sample_size
    
#     # --- ROO Method ---
#     residuals_ROO = []
#     for i in range(sample_size):
#         # Repeat-one-observation mean
#         y_mean_ROO = (np.sum(y) + y[i]) / (sample_size + 1)
#         # Residual
#         residual_ROO = np.abs(y[i] - y_mean_ROO)
#         residuals_ROO.append(residual_ROO)
    
#     # Compute the (1 - alpha)-quantile of the residuals
#     q_ROO = np.quantile(residuals_ROO, 1 - alpha)
    
#     # Construct the confidence interval
#     CI_ROO = [y_mean - q_ROO, y_mean + q_ROO]
#     interval_widths_ROO.append(CI_ROO[1] - CI_ROO[0])
    
#     # Check coverage on a hold-out sample
#     sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#     coverage_ROO += np.sum((sample >= CI_ROO[0]) & (sample <= CI_ROO[1])) / sample_size
    
#     # --- Full Conformal Method ---
#     # Create grid of possible y values
#     # y_grid = np.linspace(y_mean - grid_width*y_std, 
#     #                     y_mean + grid_width*y_std, 
#     #                     n_grid)
    
#     y_grid = y

#     # Function to compute conformity scores
#     def compute_conformity_scores(y_aug):
#         """Compute conformity scores for augmented dataset"""
#         mean_aug = np.mean(y_aug)
#         return np.abs(y_aug - mean_aug)
#         # ROO+ 
#     def max_quantile(vs, alpha=0.025):
#         # The \ceil(1-alpha)(n+1) largest value of vs
#         vs_ordered = np.sort(vs)
#         return vs_ordered[-1 +int(np.ceil((1-alpha)*(sample_size+1)))]


#     def min_quantile(vs, alpha=0.025):
#         # The floor(alpha(n+1))-th smallest value of vs
#         vs_ordered = np.sort(vs)
#         return vs_ordered[-1 + int(np.floor(alpha*(sample_size+1)))]

#     # Storage for conformity scores for each grid point
#     grid_scores = []
#     valid_points = []
#     include_trail_list = []

#     L_lower_set = []
#     valid_points_roo = []
#     L_upper_set = []
#     interval_widths_ROOP = []

#     # For each potential y value
#     for y_trial in y:
#         # Augment the dataset
#         y_aug = np.append(y, y_trial)
        
#         # Compute conformity scores
#         scores = compute_conformity_scores(y_aug)
        
#         # Store the score for the trial value
#         # grid_scores.append(scores[-1])

#         # Step 1: Calculate the lhs (1-p-value)
#         lhs = np.sum((scores <= scores[-1]))

#         # Step 2: Compute the right-hand side (RHS) of your condition
#         rhs = np.ceil((sample_size + 1) * (1 - alpha))

#         # Step 3
#         include_trial = lhs <= rhs
#         include_trail_list.append(include_trial)

#         if include_trial.astype(int).tolist() == 1:
#             valid_points.append(y_trial)


#         # Roo FULL
#         # Augment the dataset
#         mean_aug = np.mean(y_aug)


#         L_lower_j = mean_aug - max_quantile(scores)
#         L_upper_j = mean_aug + max_quantile(scores)

#         if y_trial > L_lower_j and y_trial <= L_upper_j:
#             valid_points_roo.append(y_trial)

#         # L_lower_set.append(L_lower_j)
#         # L_upper_set.append(L_upper_j)
#         #___

#     if len(valid_points) > 0:
#         # Construct the confidence interval
#         CI_conformal = [np.min(valid_points), np.max(valid_points)]
#         interval_widths_conformal.append(CI_conformal[1] - CI_conformal[0])
        
#         # Check coverage on a hold-out sample
#         sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#         coverage_conformal += np.sum((sample >= CI_conformal[0]) & 
#                                     (sample <= CI_conformal[1])) / sample_size
        


#         CI_conformal_ROO = [np.min(valid_points_roo), np.max(valid_points_roo)]
#         interval_widths_ROOP.append(CI_conformal_ROO[1] - CI_conformal_ROO[0])

#         # Check coverage on a hold-out sample
#         coverage_conformal_ROO += np.sum((sample >= CI_conformal_ROO[0]) & 
#                                     (sample <= CI_conformal_ROO[1])) / sample_size
        
#         if CI_conformal_ROO[1] != CI_conformal[1]:
#             print(f"Warning: CI_conformal_ROO[1] != CI_conformal[1] in simulation {sim}")
#             print(f"Upper bound: {CI_conformal_ROO[1], CI_conformal[1]}")
#             print(f"Lower bound: {CI_conformal_ROO[0], CI_conformal[0]}")
#             break
#         elif CI_conformal_ROO[0] != CI_conformal[0]:
#             print(f"Warning: CI_conformal_ROO[0] 1= CI_conformal[0] in simulation {sim}")
#             print(f"Upper bound: {CI_conformal_ROO[1], CI_conformal[1]}")
#             print(f"Lower bound: {CI_conformal_ROO[0], CI_conformal[0]}")
#             break
#     else:
#         print(f"Warning: No valid points found in simulation {sim}")
#         # Use dummy values for this simulation
#         interval_widths_conformal.append(np.nan)


#         # L_lower_set = []
#         # L_upper_set = []
#         # interval_widths_ROOP = []
#         # for y_trial in y:
#         #     # Augment the dataset
#         #     y_aug = np.append(y, y_trial)

#         #     mean_aug = np.mean(y_aug)

#         #     scores = compute_conformity_scores(y_aug)

#         #     L_lower_j = mean_aug - max_quantile(scores)
#         #     L_upper_j = mean_aug + max_quantile(scores)

#         #     L_lower_set.append(L_lower_j)
#         #     L_upper_set.append(L_upper_j)

#         # CI_conformal = [np.max(L_lower_set), np.min(L_upper_set)]
#         # interval_widths_ROOP.append(CI_conformal[1] - CI_conformal[0])

#         # # Check coverage on a hold-out sample
#         # sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#         # coverage_ROOP += np.sum((sample >= CI_conformal[0]) & 
#         #                            (sample <= CI_conformal[1])) / sample_size






# # Calculate coverage probabilities
# coverage_prob_LOO = coverage_LOO / n_simulations
# coverage_prob_ROO = coverage_ROO / n_simulations
# coverage_prob_conformal = coverage_conformal / n_simulations
# coverage_prob_ROOP = coverage_conformal_ROO / n_simulations

# # Calculate average interval widths
# avg_interval_width_LOO = np.mean(interval_widths_LOO)
# avg_interval_width_ROO = np.mean(interval_widths_ROO)
# avg_interval_width_conformal = np.nanmean(interval_widths_conformal)
# avg_interval_width_ROOP = np.mean(interval_widths_ROOP)


# # Print the results
# print("Results after {} simulations:".format(n_simulations))
# print("\nLOO Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_LOO))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_LOO))
# print("\nROO Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_ROO))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_ROO))
# print("\nFull Conformal Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_conformal))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_conformal))
# print("\n ROO Full Conformal Method:")
# print("Coverage Probability: {:.3f}".format(coverage_prob_ROOP))
# print("Average Interval Width: {:.3f}".format(avg_interval_width_ROOP))

# # Create figure with subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # Plot 1: Distribution of interval widths
# ax1.hist(interval_widths_LOO, bins=30, alpha=0.45, label='LOO', edgecolor='blue')
# ax1.hist(interval_widths_ROO, bins=30, alpha=0.45, label='ROO', edgecolor='orange')
# ax1.hist(interval_widths_conformal, bins=30, alpha=0.45, label='Conformal', edgecolor='green')
# ax1.hist(interval_widths_ROOP, bins=30, alpha=0.45, label='ROO Full Conformal', edgecolor='red')
# ax1.set_xlabel('Interval Width')
# ax1.set_ylabel('Frequency')
# ax1.set_title('Distribution of Confidence Interval Widths')
# ax1.legend()

# # Plot 2: Coverage comparison
# methods = ['LOO', 'ROO', 'Conformal', 'ROO Full Conformal']
# coverages = [coverage_prob_LOO, coverage_prob_ROO, coverage_prob_conformal, coverage_prob_ROOP]
# widths = [avg_interval_width_LOO, avg_interval_width_ROO, avg_interval_width_conformal, avg_interval_width_ROOP]

# x = np.arange(len(methods))
# width = 0.35

# ax2.bar(x - width/2, coverages, width, label='Coverage')
# ax2.bar(x + width/2, widths, width, label='Avg Width')
# ax2.axhline(y=1-alpha, color='r', linestyle='--', label='Target Coverage')
# ax2.set_ylabel('Probability / Width')
# ax2.set_title('Coverage Probability and Average Interval Width')
# ax2.set_xticks(x)
# ax2.set_xticklabels(methods)
# ax2.legend()

# plt.tight_layout()
# plt.show()
# # %%
# # --- Full Conformal Method ---
# # Create grid of possible y values
# # y_grid = np.linspace(y_mean - grid_width*y_std, 
# #                     y_mean + grid_width*y_std, 
# #                     n_grid)
# y_grid = y

# # Function to compute conformity scores
# def compute_conformity_scores(y_aug):
#     """Compute conformity scores for augmented dataset"""
#     mean_aug = np.mean(y_aug)
#     return np.abs(y_aug - mean_aug)
#     # ROO+ 
# def max_quantile(vs, alpha=0.05):
#     # The \ceil(1-alpha)(n+1) largest value of vs
#     vs_ordered = np.sort(vs)
#     return vs_ordered[-2 + int(np.ceil((1-alpha)*(sample_size+1)))]


# def min_quantile(vs, alpha=0.05):
#     # The floor(alpha(n+1))-th smallest value of vs
#     vs_ordered = np.sort(vs)
#     return vs_ordered[-1 + int(np.floor(alpha*(sample_size+1)))]

# # Storage for conformity scores for each grid point
# grid_scores = []
# valid_points = []
# include_trail_list = []

# L_lower_set = []
# valid_points_roo = []
# L_upper_set = []
# interval_widths_ROOP = []

# # For each potential y value
# for y_trial in y:
#     # Augment the dataset
#     y_aug = np.append(y, y_trial)
    
#     # Compute conformity scores
#     scores = compute_conformity_scores(y_aug)
    
#     # Store the score for the trial value
#     # grid_scores.append(scores[-1])

#     # Step 1: Calculate the lhs (1-p-value)
#     lhs = np.sum((scores <= scores[-1]))

#     # Step 2: Compute the right-hand side (RHS) of your condition
#     rhs = np.ceil((sample_size + 1) * (1 - alpha))

#     # Step 3
#     include_trial = lhs <= rhs
#     include_trail_list.append(include_trial)

#     if include_trial.astype(int).tolist() == 1:
#         valid_points.append(y_trial)


#     # Roo FULL
#     # Augment the dataset
#     mean_aug = np.mean(y_aug)


#     L_lower_j = mean_aug - max_quantile(scores)
#     L_upper_j = mean_aug + max_quantile(scores)

#     if y_trial > L_lower_j and y_trial < L_upper_j:
#         valid_points_roo.append(y_trial)

#     # L_lower_set.append(L_lower_j)
#     # L_upper_set.append(L_upper_j)
#     #___

# if len(valid_points) > 0:
#     # Construct the confidence interval
#     CI_conformal = [np.min(valid_points), np.max(valid_points)]
#     interval_widths_conformal.append(CI_conformal[1] - CI_conformal[0])
    
#     # Check coverage on a hold-out sample
#     sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
#     coverage_conformal += np.sum((sample >= CI_conformal[0]) & 
#                                 (sample <= CI_conformal[1])) / sample_size
    


#     # CI_conformal_ROO = [np.max(L_lower_set), np.min(L_upper_set)]
#     # interval_widths_ROOP.append(CI_conformal_ROO[1] - CI_conformal_ROO[0])

#     # # Check coverage on a hold-out sample
#     # coverage_ROOP += np.sum((sample >= CI_conformal_ROO[0]) & 
#     #                         (sample <= CI_conformal_ROO[1])) / sample_size
# else:
#     print(f"Warning: No valid points found in simulation {sim}")
#     # Use dummy values for this simulation
#     interval_widths_conformal.append(np.nan)

# # %%
# print(CI_conformal_ROO)
# print(CI_conformal)

# # %%
# np.max(valid_points)

# # %%
# np.max(valid_points_roo)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)  # For reproducibility
n_simulations = 1000  # Number of Monte Carlo simulations
sample_size = 5  # Sample size
alpha = 0.3  # Significance level (for 95% confidence intervals)
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
        print("Y_aug", y_aug)
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
        print("Y_aug", y_aug)
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

# # %%
# # Check if y_grid has values equal to any in y 
# # y_grid = np.linspace(-1, 10, 100)
# # y_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# for y_trial in y_grid:
#     if y_trial in y:
#         print(f"y_trial: {y_trial} is in y")

# # %% Exploring differences
# y = [1, 2, 3,4, 60]
# # y_trials  = np.linspace(-1, 10, 100)
# y_trials = [2]
# n = len(y)
# K = np.ceil((n+1)*(1-0.50))

# for y_trial in y_trials:
#     y_aug = np.append(y, y_trial)

#     y_aug_mean = np.mean(y_aug)
#     scores = compute_conformity_scores(y_aug)

#     # Compute the (1 - alpha)-quantile of the conformity scores
#     q = max_quantile(scores, alpha=0.5)

    
#     # Check if y_trial is included in y_aug_mean +/- q
#     roo_trial = y_aug_mean - q <= y_trial <= y_aug_mean + q
     
#     # If rank of scores[-1] <= K, then include y_trial in the prediction set
#     CP_trial = np.sum(scores <= scores[-1]) <= K

#     print(f"y_trial: {y_trial}, roo_trial: {roo_trial}, CP_trial: {CP_trial}")

#     # Check if roo_trial and CP_trial differ. 
#     if roo_trial != CP_trial:
#         print(f"!!!!y_trial: {y_trial}, roo_trial: {roo_trial}, CP_trial: {CP_trial}")
#         # break
    
# %%
# import numpy as np

# # Simulation parameters
# np.random.seed(0)  # For reproducibility
# n_simulations = 1_000  # Number of Monte Carlo simulations
# sample_size = 10  # Sample size
# alpha = 0.05  # Significance level (for 95% confidence intervals)
# true_mean = 0  # True mean of the distribution
# true_std = 1  # True standard deviation
# # Grid parameters for full conformal
# n_grid = 1000  # Number of grid points
# grid_width = 4  # Number of standard deviations to extend grid

# # Storage for results
# coverage_conformal = 0
# coverage_conformal_ROO = 0
# interval_widths_conformal = []
# interval_widths_ROOP = []

# def max_quantile(vs, alpha=0.05):
#     """Compute the (1 - alpha)-quantile for conformity scores."""
#     vs_ordered = np.sort(vs)
#     return vs_ordered[int(np.ceil((1 - alpha) * len(vs))) - 1]

# def randomized_inclusion_decision(scores, trial_score, K):
#     """
#     Make a randomized decision for including the trial value when there are ties.
    
#     Args:
#         scores: Array of all conformity scores including trial score
#         trial_score: The conformity score of the trial value
#         K: The rank threshold
    
#     Returns:
#         Boolean indicating whether to include the trial value
#     """
#     # Count scores strictly less than trial_score
#     n_less = np.sum(scores < trial_score)
    
#     # Count number of ties (including trial score)
#     n_ties = np.sum(scores == trial_score)
    
#     # If strictly less than K scores are smaller, always include
#     if n_less <= K:
#         # Calculate how many slots are left at the threshold
#         slots_left = K - n_less
        
#         # If there are ties, randomly include based on proportion of slots left
#         if n_ties > 0:
#             p_include = slots_left / n_ties
#             # p_include = 1
#             return np.random.uniform() < p_include
    
#     return False

# def compute_conformity_scores(y_aug, 
#                               mean_aug):
#     """Compute conformity scores for augmented dataset."""
#     return np.abs(y_aug - mean_aug)

# for sim in range(n_simulations):
#     # Generate a random sample from a normal distribution
#     y = - np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    
#     # Estimate the mean
#     y_mean = np.mean(y)
#     y_std = np.std(y)
    
#     # Create a dense grid of possible y values
#     y_grid = np.linspace(
#         y_mean - grid_width * y_std,
#         y_mean + grid_width * y_std,
#         n_grid
#     )
    
#     valid_points = []
#     valid_points_roo = []
    
#     # Calculate the threshold
#     K = int(np.ceil((sample_size + 1) * (1 - alpha)))
    
#     for y_trial in y_grid:
#         # Augment the dataset with the trial value
#         y_aug = np.append(y, y_trial)
        
#         # Compute conformity scores
#         y_aug_mean  = np.mean(y_aug)
#         scores = compute_conformity_scores(y_aug, mean_aug = y_aug_mean)
        
#         # ---------------- Full Conformal with Randomization ----------------
#         # Use randomized inclusion decision for ties
#         include_trial = randomized_inclusion_decision(scores, scores[-1], K)
        
#         if include_trial:
#             valid_points.append(y_trial)
            
#         # ------------------- Conformal ROO (ROOP) -------------------
#         # Compute the (1 - alpha)-quantile of the conformity scores
#         q = max_quantile(scores, alpha=alpha)
        
#         # Compute the augmented mean
#         mean_aug = np.mean(y_aug)
        
#         # Define the prediction interval based on ROOP
#         L_lower_j = mean_aug - q
#         L_upper_j = mean_aug + q
        
#         # Check if the trial value falls within the ROOP interval
#         if L_lower_j <= y_trial <= L_upper_j:
#             valid_points_roo.append(y_trial)
    
#     test_sample = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
    
#     # Construct the conformal confidence interval
#     if valid_points:
#         CI_conformal = [np.min(valid_points), np.max(valid_points)]
#         interval_widths_conformal.append(CI_conformal[1] - CI_conformal[0])
#         # Check coverage on a hold-out sample
#         coverage_conformal += np.mean((test_sample >= CI_conformal[0]) & 
#                                     (test_sample <= CI_conformal[1]))
#     else:
#         interval_widths_conformal.append(np.nan)
    
#     # Construct the ROOP confidence interval
#     if valid_points_roo:
#         CI_conformal_ROO = [np.min(valid_points_roo), np.max(valid_points_roo)]
#         interval_widths_ROOP.append(CI_conformal_ROO[1] - CI_conformal_ROO[0])
#         # Check coverage on a hold-out sample
#         coverage_conformal_ROO += np.mean((test_sample >= CI_conformal_ROO[0]) & 
#                                         (test_sample <= CI_conformal_ROO[1]))
#     else:
#         interval_widths_ROOP.append(np.nan)


# print(CI_conformal)
# print(CI_conformal_ROO)

# # Calculate coverage probabilities
# coverage_prob_conformal = coverage_conformal / n_simulations
# coverage_prob_ROOP = coverage_conformal_ROO / n_simulations

# # Calculate average interval widths
# avg_interval_width_conformal = np.nanmean(interval_widths_conformal)
# avg_interval_width_ROOP = np.nanmean(interval_widths_ROOP)

# # Print the results
# print(f"Results after {n_simulations} simulations:")

# print("\nFull Conformal Method:")
# print(f"Coverage Probability: {coverage_prob_conformal:.3f}")
# print(f"Average Interval Width: {avg_interval_width_conformal:.3f}")

# print("\nConformal ROO (ROOP) Method:")
# print(f"Coverage Probability: {coverage_prob_ROOP:.3f}")
# print(f"Average Interval Width: {avg_interval_width_ROOP:.3f}")

# # Create figure with subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # Plot 1: Distribution of interval widths
# ax1.hist(interval_widths_conformal, bins=30, alpha=0.6, label='Conformal', edgecolor='green')
# ax1.hist(interval_widths_ROOP, bins=30, alpha=0.6, label='Conformal ROO (ROOP)', edgecolor='red')
# ax1.set_xlabel('Interval Width')
# ax1.set_ylabel('Frequency')
# ax1.set_title('Distribution of Confidence Interval Widths')
# ax1.legend()

# # Plot 2: Coverage comparison
# methods = ['Conformal', 'Conformal ROO (ROOP)']
# coverages = [coverage_prob_conformal, coverage_prob_ROOP]
# widths = [avg_interval_width_conformal, avg_interval_width_ROOP]

# x = np.arange(len(methods))
# width_bar = 0.35

# # Create bar plots for coverage and average width
# ax2.bar(x - width_bar/2, coverages, width_bar, label='Coverage Probability', color='skyblue')
# ax2.bar(x + width_bar/2, widths, width_bar, label='Average Interval Width', color='salmon')
# ax2.axhline(y=1 - alpha, color='gray', linestyle='--', label='Target Coverage')
# ax2.set_ylabel('Probability / Width')
# ax2.set_title('Coverage Probability and Average Interval Width')
# ax2.set_xticks(x)
# ax2.set_xticklabels(methods)
# ax2.legend()

# plt.tight_layout()
# plt.show()
# %%
alpha = 0.05
y = np.array([ 1.46564877, -0.2257763 ,  0.0675282 , -1.42474819, -0.54438272,
        0.11092259, -1.15099358,  0.37569802, -0.60063869, -0.29169375])
trial = -3.31036857

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


# %%
q = max_quantile(scores, alpha=alpha)
print(q)

# Compute the augmented mean
mean_aug = np.mean(y_aug)

# Define the prediction interval based on ROOP
L_lower_j = mean_aug - q
print(f'lower: {L_lower_j}')
L_upper_j = mean_aug + q
print(f"upper: {L_upper_j}")


# %%
