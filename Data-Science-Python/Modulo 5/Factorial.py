## LOAD PACKAGES
import pandas as pd
import numpy as np
import scipy.stats as ss
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

## Load the dataset
dat = pd.read_csv("datasets/logos.csv")

## Remove rows with missing values
dat.dropna(subset = ['logo'], inplace = True)

## Compute sentiment and look at the head of the data frame.
dat['sentiment'] = dat[['friendly', 'inviting', 'interesting', 'positive', 'pleasant']].apply(np.mean, axis = 1)
print(dat.head())


ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.boxplot(x = 'sex', y = 'sentiment', data = dat, ax = ax)
sns.swarmplot(x = 'sex', y = 'sentiment', color = 'black', data = dat, ax = ax, alpha = 0.4)

ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.boxplot(x = 'logo', y = 'sentiment', data = dat, hue = 'sex', ax = ax)
sns.swarmplot(x = 'logo', y = 'sentiment', hue = 'sex', data = dat, ax = ax, alpha = 0.4)

dat_grouped = dat[['sentiment','logo','sex']].groupby(['logo','sex'])
print('The means of the groups:')
print(dat_grouped.mean())
print('\n')
print('The standard deviations of the groups:')
print(dat_grouped.std())


formula = 'sentiment ~ C(logo) + C(sex) + C(logo):C(sex)'
lm_model = ols(formula, dat).fit()
aov_table = anova_lm(lm_model, typ=2)
print(aov_table)

dat['logo_sex'] = [x.replace(" ", "") + '_' + y for x,y in zip(dat.logo,dat.sex)]

# Run the Tukey HDS test using the interaction variable and display the results
Tukey_HSD = pairwise_tukeyhsd(dat.sentiment,dat.logo_sex)
print(Tukey_HSD)
Tukey_HSD.plot_simultaneous()
plt.show()

plt.show()