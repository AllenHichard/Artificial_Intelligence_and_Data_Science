#### LOAD DATA ####
import pandas as pd
dat = pd.read_csv("datasets/validity.csv", index_col = 'Unnamed: 0')
print(dat.dtypes)
print(dat.head())


#correlations
corr_mat = dat.corr().round(2)
print(corr_mat)

import numpy as np
import scipy.stats as ss
import math

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

def print_cis(corr_mat,var1, var2, idx1, idx2):
    print('\nFor ' + var1 + ' vs. ' + var2)
    conf_ints = r_conf_int(corr_mat[idx1,idx2], 0.05, 1000)
    print('Correlation = %4.3f with CI of %4.3f to %4.3f' % (corr_mat[idx1,idx2], conf_ints[0], conf_ints[1]))

corr_mat = np.array(corr_mat)

print_cis(corr_mat, 'sent', 'WC', 1, 0)
print_cis(corr_mat, 'sent', 'rating', 0, 2)
print_cis(corr_mat, 'sent', 'purchase', 0, 3)
