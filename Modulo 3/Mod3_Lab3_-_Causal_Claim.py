#### LOAD PACKAGES ####
from scipy import stats
import scipy.stats as ss
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as ws
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline


#### LOAD DATA ####
import pandas as pd
dat = pd.read_csv("datasets/causal.csv")

# Inspect data
print(dat.columns)

print('\n')
print(dat.head())

print('\n')
print(dat.group.unique())

print(dat.describe())

print(dat[['group','prod']].groupby('group').mean())
print(dat[['group','prod']].groupby('group').std())

"""
O código Python na função abaixo faz o seguinte:

Calcule a diferença de médias.
A função ttest_ind do pacote scipy.stats para calcular a estatística t e o valor p.
A função tconfint_diff é usada para calcular o intervalo de confiança.
A função dof_satt estima os graus de liberdade.
"""


def t_test_two_samp(df, alpha, alternative='two-sided'):
    a = df[df.group == 'control']['prod']
    b = df[df.group == 'intervention']['prod']

    diff = a.mean() - b.mean()

    res = ss.ttest_ind(a, b)

    means = ws.CompareMeans(ws.DescrStatsW(a), ws.DescrStatsW(b))
    confint = means.tconfint_diff(alpha=alpha, alternative=alternative, usevar='unequal')
    degfree = means.dof_satt()

    index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
    return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index=index)


test = t_test_two_samp(dat, 0.05)
print(test)
ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.boxplot(x = 'group', y = 'prod', data = dat, ax = ax)
sns.swarmplot(x = 'group', y = 'prod', color = 'black', data = dat, ax = ax, alpha = 0.4)


ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.violinplot(x = 'group', y = 'prod', data = dat, ax = ax)
sns.swarmplot(x = 'group', y = 'prod', color = 'black', data = dat, ax = ax, alpha = 0.4)

#cohen.d(prod ~ group, data=dat)
control = dat[dat.group == 'control']['prod']
intervention = dat[dat.group == 'intervention']['prod']
print(np.mean(intervention) - np.mean(control))
ratio = len(control)/len(intervention)
tt_ind_solve_power(effect_size=None, nobs1 = len(control), alpha=0.05, power=0.8, ratio=ratio, alternative='two-sided')