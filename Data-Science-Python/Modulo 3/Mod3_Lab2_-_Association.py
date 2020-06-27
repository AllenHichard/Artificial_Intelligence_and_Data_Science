
#### LOAD PACKAGES ####
## Use inline magic command so plots appear in the data frame
#%matplotlib inline

## Next the packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import math

#### LOAD DATA ####
dat = pd.read_csv("datasets/loyaltydata.csv")
print(dat.columns)

dat.head()
print(dat.describe())
print(dat[['loyalty1', 'loyalty2', 'loyalty3']].corr())
corr_mat = dat[['loyalty1', 'loyalty2', 'loyalty3']].corr().round(2)
print(corr_mat)

"""
| # | Correlation |  Meaning   |
|---|-------------|------------|
| 1 |  0.0 - 0.1  | Negligible |
| 2 |  0.1 - 0.3  |   Small    |
| 3 |  0.3 - 0.5  |   Medium   |
| 4 |    0.5 +    |   Large    |
"""

sns.heatmap(corr_mat, vmax=1.0)
plt.title('Correlation matrix for loyalty features')
plt.yticks(rotation='horizontal')
plt.xticks(rotation='vertical')
#plt.show()
a = sns.lmplot("loyalty1", "loyalty2", dat, x_jitter=.15, y_jitter=.15, scatter_kws={'alpha':0.2}, fit_reg = False)
b = sns.lmplot("loyalty1", "loyalty3", dat, x_jitter=.15, y_jitter=.15, scatter_kws={'alpha':0.2}, fit_reg = False)
c = sns.lmplot("loyalty2", "loyalty3", dat, x_jitter=.15, y_jitter=.15, scatter_kws={'alpha':0.2}, fit_reg = False)
print(a,b,c)

def r_z(r):
    return math.log((1 + r) / (1 - r)) / 2.0

def z_r(z):
    e = math.exp(2 * z)
    return((e - 1) / (e + 1))

def r_conf_int(r, alpha, n):
    # Transform r to z space
    z = r_z(r)
    # Compute standard error and critcal value in z
    se = 1.0 / math.sqrt(n - 3)
    z_crit = ss.norm.ppf(1 - alpha/2)

    ## Compute CIs with transform to r
    lo = z_r(z - z_crit * se)
    hi = z_r(z + z_crit * se)
    return (lo, hi)

print('\nFor loyalty1 vs. loyalty2')
corr_mat = np.array(corr_mat)
conf_ints = r_conf_int(corr_mat[1,0], 0.05, 1000)
print('Correlation = %4.3f with CI of %4.3f to %4.3f' % (corr_mat[1,0], conf_ints[0], conf_ints[1]))
print('\nFor loyalty1 vs. loyalty3')
conf_ints = r_conf_int(corr_mat[2,0], 0.05, 1000)
print('Correlation = %4.3f with CI of %4.3f to %4.3f' % (corr_mat[2,0], conf_ints[0], conf_ints[1]))
print('\nFor loyalty2 vs. loyalty3')
conf_ints = r_conf_int(corr_mat[2,1], 0.05, 1000)
print('Correlation = %4.3f with CI of %4.3f to %4.3f' % (corr_mat[2,1], conf_ints[0], conf_ints[1]))