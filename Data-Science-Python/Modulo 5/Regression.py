#### LOAD PACKAGES
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline


#### LOAD DATA ####
dat = pd.read_csv("datasets/regionalhappy.csv")
print(dat.columns)
dat.columns = ["Happiness", "GDP", "Family", "Life.Expect", "Freedom", "Generosity", "Trust.Gov", "Dystopia"]
dat.head()

## Add a constant term to the array of predictors to get an intercept term
predictors = sm.add_constant(dat.GDP, prepend = False)
lm_mod = sm.OLS(dat.Happiness, predictors)
res = lm_mod.fit()
print(res.summary()) #Happiness' = 3.2 + 2.18(GDP).

print('Range of Happiness = { ' + str(min(dat.Happiness)) + ' ' + str(max(dat.Happiness)) + '}')

ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.Happiness.plot.hist(ax = ax, alpha = 0.6)
plt.title('Histogram of Happiness')
plt.xlabel('Happiness')


ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.GDP.plot.hist(ax = ax, alpha = 0.6)
plt.title('Histogram of GDP')
plt.xlabel('GDP')



## Create a new data frame with the predictor value and the constant
new_predict = pd.DataFrame({'GDP':[0.5], 'const':[1.0]})
## Make prediction with new values
print(res.predict(new_predict))

## Create a new data frame with the predictor value and the constant
new_predict = pd.DataFrame({'GDP':[0.5, 0.9, 1.0]})
new_predict = sm.add_constant(new_predict, prepend = False)
## Make prediction with new values
print(res.predict(new_predict))

ax = plt.figure(figsize=(8, 8)).gca() # define axis
sns.regplot(x="GDP", y="Happiness", data=dat, ax = ax)
plt.title('Happiness vs. GDP with linear regression line')

print(res.summary())



## Create a new data frame with the predictor value and the constant
new_predict = dat.GDP
new_predict = sm.add_constant(new_predict, prepend = False)
## Make prediction with new values
new_predict['Score'] = res.predict(new_predict)
## Compute the residuals
new_predict['Residuals'] = dat.Happiness - new_predict.Score

ax = plt.figure(figsize=(8, 8)).gca() # define axis
new_predict.plot.scatter(x='Score',y='Residuals', ax = ax)
plt.title('Residuals vs. predicted value')
plt.ylabel("Residual values")
plt.xlabel('Predicted values')


ax = plt.figure(figsize=(8, 6)).gca() # define axis
new_predict.Residuals.plot.hist(ax = ax)
#plt.show()



## Add a constant term to the array of predictors to get an intercept term
predictors = sm.add_constant(dat[['GDP','Freedom']], prepend = False)

lm_mod_2 = sm.OLS(dat.Happiness, predictors)
res_2 = lm_mod_2.fit()
print(res_2.summary()) #Happiness' = 2.55 + 1.87(GDP) + 2.36(Freedom).
print(anova_lm(res, res_2, typ = 1))

print('Adjusted coefficient for GDP = ' + str(res_2.params[0] * np.std(dat.GDP)/np.std(dat.Happiness)))
print('Adjusted coefficient for Freedom = ' + str(res_2.params[1] * np.std(dat.Freedom)/np.std(dat.Happiness)))


## Create a new data frame with the predictor values and the constant
new_predict = dat[['GDP','Freedom']]
new_predict = sm.add_constant(new_predict, prepend = False)
## Make prediction with new values and 2 predictor model
new_predict['Score'] = res_2.predict(new_predict)
## Compute the residuals
new_predict['Residuals'] = dat.Happiness - new_predict.Score

## Plot the histogram of the residuals
ax = plt.figure(figsize=(8, 6)).gca() # define axis
new_predict.Residuals.plot.hist(ax = ax)
#plt.show()
print(new_predict['Residuals'].mean())