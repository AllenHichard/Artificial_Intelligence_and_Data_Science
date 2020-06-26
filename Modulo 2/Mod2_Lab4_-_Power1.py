### LOAD PACKAGES ####
from scipy import stats
""" , d de Cohen,
| #  |    d Value    |  Meaning   |
|:--:|:-------------:|:----------:|
| 1. |    0 - 0.2    | Negligible |
| 2. |   0.2 - 0.5   |   Small    |
| 3. |   0.5 - 0.8   |   Medium   |
| 4. |     0.80 +    |   Large    |
"""
from statsmodels.stats.power import tt_ind_solve_power
a = tt_ind_solve_power(effect_size=0.4, nobs1 = 40, alpha=0.05, power=None, ratio=1, alternative='two-sided')
b = tt_ind_solve_power(effect_size=0.4, nobs1 = 100, alpha=0.05, power=None, ratio=1, alternative='two-sided')
c = tt_ind_solve_power(effect_size=0.4, nobs1 = 150, alpha=0.05, power=None, ratio=1, alternative='two-sided')
print(a, b, c)

print(tt_ind_solve_power(effect_size=0.4, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided'))
print(tt_ind_solve_power(effect_size=0.2, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided'))
print(tt_ind_solve_power(effect_size=0.8, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided'))

print(tt_ind_solve_power(effect_size=0.3, nobs1 = 26, alpha=0.05, power=None, ratio=1, alternative='two-sided'))

import pandas as pd
d_vals = [x/10.0 for x in range(1, 16)]
powers = [tt_ind_solve_power(effect_size=x, nobs1 = 26, alpha=0.05, power=None, ratio=1, alternative='two-sided')
                    for x in d_vals]
d_powers = pd.DataFrame({'d_values':d_vals, 'power':powers})
print(d_powers)

## create list of d values
d_vals = [x / 10.0 for x in range(2, 16)]

## Initialize data frame
powers = pd.DataFrame({'sample_size': range(20, 210, 10)})
## Loop over d values
for d_val in d_vals:
    col_name = 'd = ' + str(d_val)
    ## List comprehension for each d value itterating over the sample sizes
    powers[col_name] = [
        tt_ind_solve_power(effect_size=d_val, nobs1=x, alpha=0.05, power=None, ratio=1, alternative='two-sided')
        for x in range(20, 210, 10)]

print(powers)


import matplotlib.pyplot as plt
#%matplotlib inline
fig = plt.figure(figsize=(12,10)) # define plot area
ax = fig.gca() # define axis
powers.plot(x = 'sample_size', ax = ax, linestyle = '-.')
plt.hlines(y = 0.8, xmin = 20, xmax = 200, color = 'red', linestyle = '--')
plt.title('Power vs. sample size for values of d')
plt.ylabel('Power')
plt.xlabel('Sample Size')
plt.show()