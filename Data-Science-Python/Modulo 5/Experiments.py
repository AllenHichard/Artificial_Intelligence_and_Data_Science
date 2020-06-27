#### LOAD PACKAGES ####
import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.stats.weightstats as ws
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

#### LOAD DATA ####
dat = pd.read_csv("datasets/logos.csv")
print(dat.columns)
print(dat.head())
print(dat.describe())

print(dat[['friendly', 'inviting', 'interesting', 'positive', 'pleasant']].corr().round(3))

dat['sentiment'] = dat[['friendly', 'inviting', 'interesting', 'positive', 'pleasant']].apply(np.mean, axis = 1)
print(dat.head())

## Print some summary statistics
print('Mean of Sentiment = ' + str(np.mean(dat.sentiment)))
print('STD of Sentiment = ' + str(np.std(dat.sentiment)))

## Plot a histogram
ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.sentiment.plot.hist(ax = ax, alpha = 0.6, bins = 15)
plt.title('Histogram of Sentiment')
plt.xlabel('Sentiment')

for col in dat.columns:
    print(col + ' has missing values ' +
          str((dat[col].isnull().values.any())) or str(dat[col].isna().values.any()))

print(dat.shape)
dat.dropna(subset = ['logo'], inplace = True)
print(dat.shape)

## Check once more
print('\n')
for col in dat.columns:
    print(col + ' has missing values ' +
          str((dat[col].isnull().values.any())) or str(dat[col].isna().values.any()))

ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.boxplot(x = 'logo', y = 'sentiment', data = dat, ax = ax)
sns.swarmplot(x = 'logo', y = 'sentiment', color = 'black', data = dat, ax = ax, alpha = 0.4)


logo_grouped = dat[['logo','sentiment']].groupby('logo')
print(' Mean by logo')
print(logo_grouped.mean().round(2))
print('\n Standard deviation by logo')
print(logo_grouped.std().round(2))


def t_test_two_samp(a, b, alpha, alternative='two-sided'):
    diff = a.mean() - b.mean()

    res = ss.ttest_ind(a, b)

    means = ws.CompareMeans(ws.DescrStatsW(a), ws.DescrStatsW(b))
    confint = means.tconfint_diff(alpha=alpha, alternative=alternative, usevar='unequal')
    degfree = means.dof_satt()

    index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
    return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index=index)


test = t_test_two_samp(dat.loc[dat.logo == 'Logo A', 'sentiment'], dat.loc[dat.logo == 'Logo B', 'sentiment'], 0.05)
print(test)

"""
	d Value	Meaning
1.	0 - 0.2	Negligible
2.	0.2 - 0.5	Small
3.	0.5 - 0.8	Medium
4.	0.80 +	Large
"""

print(logo_grouped.count())
d = (8.58 - 8.44)/(np.std(dat.loc[dat.logo == 'Logo A', 'sentiment']))
print("d = ", d)
print("power = ", tt_ind_solve_power(effect_size=d, nobs1 = 32, alpha=0.05, power=None, ratio=1, alternative='two-sided'))

ax = plt.figure(figsize=(8,8)).gca() # define axis
temp = dat[dat.logo != 'Logo C']
sns.boxplot(x = 'logo', y = 'sentiment', data = temp, ax = ax)
sns.swarmplot(x = 'logo', y = 'sentiment', color = 'black', data = temp, ax = ax, alpha = 0.4)
#plt.show()


print("efeito = ", tt_ind_solve_power(effect_size=None, nobs1 = 32, alpha=0.05, power=0.8, ratio=1, alternative='two-sided'))
print("N amostras = ", tt_ind_solve_power(effect_size=0.2, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided'))

f_statistic, p_value = ss.f_oneway(dat.loc[dat.logo == 'Logo A', 'sentiment'],
                                   dat.loc[dat.logo == 'Logo B', 'sentiment'],
                                   dat.loc[dat.logo == 'Logo C', 'sentiment'])
print('F-Satatistic = ' + str(f_statistic))
print('p_value = ' + str(p_value))

Tukey_HSD = pairwise_tukeyhsd(dat.sentiment, dat.logo)
print(Tukey_HSD)

Tukey_HSD.plot_simultaneous()
#plt.show()


logo_grouped = dat[['logo','sentiment']].groupby('logo')

print(' Mean by logo')
print(logo_grouped.mean().round(2))
print('\n Standard deviation by logo')
print(logo_grouped.std().round(2))