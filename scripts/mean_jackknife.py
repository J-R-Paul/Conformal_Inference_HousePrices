# %%
import pandas as pd 
import numpy as np

# %%
# ys = np.array([1,1,2,1, 10])
n = 100
ys = np.random.normal(size=n) + 100*np.random.binomial(n=1, p=0.1, size=n)
means = np.zeros(n)
sample_mean = np.sum(ys)/n
Rloos = np.zeros(n)

for i in range(n):
    # Mean excluding the i-th value
    ys_not_i = ys[np.arange(n) != i]
    means[i] = np.sum(ys_not_i)/(n-1)
    # jackknife
    Rloos[i] = np.abs(means[i] - ys[i])


# %%
means


# %%
Rloos


# %%
jKp_lb = np.zeros(n)
jK_lb = np.zeros(n)
jkp_ub = np.zeros(n)
jk_ub = np.zeros(n)
for i in range(n):
    jKp_lb[i] = means[i] - Rloos[i]
    jK_lb[i] = sample_mean - Rloos[i]

    jkp_ub[i] = means[i] + Rloos[i]
    jk_ub[i] = sample_mean + Rloos[i]



# %%
alpha = 0.05
jkp_uq = np.quantile(jkp_ub, 1-alpha)
jkp_lq = np.quantile(jKp_lb, alpha)
print(f"JackKnife +: Lower bound: {jkp_lq}. Upper bound: {jkp_uq}")
print(f"Lenght: {jkp_uq - jkp_lq}")
# np.quantile(jK_lb, alpha)
# print(np.sort(jKp_lb))
# print(np.sort(jkp_ub))

jk_uq = np.quantile(jk_ub, 1-alpha)
jk_lq = np.quantile(jK_lb, alpha)
print(f"JackKnife -: Lower bound: {jk_lq}. Upper bound: {jk_uq}")
print(f"Lenght: {jk_uq - jk_lq}" )


# print(np.sort(jK_lb))
# print(np.sort(jk_ub))
# %%
0.1 + 0.2
# %%
