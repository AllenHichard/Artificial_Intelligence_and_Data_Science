## Load packages
import pandas as pd
import numpy as np
import scipy.stats as ss
import math
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#### LOAD DATA ####
dat = pd. read_csv("datasets/regionalhappy.csv")
print(dat.columns)

dat.columns = ["Happiness", "GDP", "Family", "Life_Expect", "Freedom", "Generosity", "Trust_Gov", "Dystopia"]
print(dat.columns)
print(dat.head())

corr_mat = dat[['Happiness', 'Life_Expect']].corr()
print(corr_mat.iloc[1,0].round(3))

"""
	Correlation	Meaning
1.	0 - 0.1	Negligible
2.	0.1 - 0.3	Small
3.	0.3 - 0.5	Medium
4.	0.50 +	Large
"""


ax = plt.figure(figsize=(8, 8)).gca() # define axis
dat.plot.scatter(x='Happiness', y='Life_Expect', ax=ax, alpha=0.4)
#plt.show()


def r_z(r):  ## transform distribution
    return math.log((1 + r) / (1 - r)) / 2.0


def z_r(z):  ## inverse transform distribution
    e = math.exp(2 * z)
    return ((e - 1) / (e + 1))


def r_conf_int(r, alpha, n):
    # Transform r to z space
    z = r_z(r)
    # Compute standard error and critcal value in z
    se = 1.0 / math.sqrt(n - 3)
    z_crit = ss.norm.ppf(1 - alpha / 2)

    ## Compute CIs with transform to r
    lo = z_r(z - z_crit * se)
    hi = z_r(z + z_crit * se)
    return (lo, hi)


def correlation_sig(df, col1, col2):
    pearson_cor = ss.pearsonr(x=df[col1], y=dat[col2])
    conf_ints = r_conf_int(pearson_cor[0], 0.05, 1000)
    print('Correlation = %4.3f with CI of %4.3f to %4.3f and p_value %4.3e'
          % (pearson_cor[0], conf_ints[0], conf_ints[1], pearson_cor[1]))


correlation_sig(dat, 'Happiness', 'Life_Expect')

ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.Happiness.plot.hist(ax = ax, alpha = 0.4)
plt.title('Histogram of Happiness')
plt.xlabel('Happiness')
#plt.show()

ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.Life_Expect.plot.hist(ax = ax, alpha = 0.4)
plt.title('Histogram of Life Expectancy')
plt.xlabel('Life Expectancy')
#plt.show()


skew = ss.skewtest(dat.Happiness)
print(skew)

dat['Life_Expect2']= max(dat.Life_Expect) + 1 - dat.Life_Expect

ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.Life_Expect2.plot.hist(ax = ax, alpha = 0.4)
plt.title('Histogram of reversed Life Expectancy')
plt.xlabel('Reversed Life Expectancy')


## Square the reversed variable
dat['Life_Expect2_sqrt'] = dat.Life_Expect2.apply(math.sqrt)

ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.Life_Expect2_sqrt.plot.hist(ax = ax, alpha = 0.4)
plt.title('Histogram of reversed Life Expectancy squared')
plt.xlabel('Reversed Life Expectancy squared')

## un-reverse the variable
dat['Life_Expect2']= max(dat.Life_Expect2_sqrt) + 1 - dat.Life_Expect2_sqrt

## compute correlation of new variable and test statistics
correlation_sig(dat, 'Happiness', 'Life_Expect2')

print(dat[['Happiness', 'Life_Expect', 'Generosity']].corr())

## Compute the correlation matrix
corrs = dat.drop(['Life_Expect2', 'Life_Expect2_sqrt'], axis = 1).corr()

## Create the hierarchical clustering model
dist = sch.distance.pdist(corrs)   # vector of pairwise distances using correlations
linkage = sch.linkage(dist, method='complete') # Compute the linkages for the clusters
ind = sch.fcluster(linkage, 0.5*dist.max(), 'distance')  # Apply the clustering algorithm

## Order the columns of the correlaton matrix according to the hierarchy
columns = [corrs.columns.tolist()[i] for i in list((np.argsort(ind)))]  # Order the names for the result
corrs_clustered = corrs.reindex(columns) ## Reindex the columns following the heirarchy

## Plot a heat map of the clustered correlations
sns.heatmap(corrs_clustered,
            xticklabels=corrs_clustered.columns.values,
            yticklabels=corrs_clustered.columns.values)

corrs_clustered.style.background_gradient().set_precision(2)

def correlation_sig2(df, col1, col2):
    pearson_cor = ss.pearsonr(x = df[col1], y = dat[col2])
    conf_ints = r_conf_int(pearson_cor[0], 0.05, 1000)
    print('Correlation with ' + col2 + ' = %4.3f with CI of %4.3f to %4.3f and p_value %4.3e'
        % (pearson_cor[0], conf_ints[0], conf_ints[1], pearson_cor[1]))

def test_significance(df, col_list):
    cols = df.columns
    for col1 in col_list:
        print('\n')
        print('Significance of correlations with ' + col1)
        for col2 in cols:
            if(col1 != col2):
                correlation_sig2(df, col1, col2)

test_cols = ['Trust_Gov', 'Generosity', 'Dystopia']
test_significance(dat, test_cols)

#plt.show()