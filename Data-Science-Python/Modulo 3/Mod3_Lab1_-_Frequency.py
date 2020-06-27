#### LOAD PACKAGES ####
## Use inline magic command so plots appear in the data frame
#%matplotlib inline

## Next the packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
import statsmodels.stats.api as sms

dat = pd.read_csv("datasets/cupsdat.csv")
print(dat.columns)

print(dat.head())
print(dat['count'].describe())
print(dat['count'].value_counts())
print(dat.shape[0])
print(len(dat['count']))

## Remove rows with nan without making copy of the data frame
dat.dropna(axis = 0, inplace = True)

## Now get the counts into a data frame sorted by the number
count_frame = dat['count'].value_counts()
count_frame = pd.DataFrame({'number':count_frame.index, 'counts':count_frame}).sort_values(by = 'number')

## Compute the percents for each number
n = len(dat['count'])
count_frame['percents'] = [100* x/n for x in count_frame['counts']]

## Print as a nice table
print(count_frame[['number', 'percents']])


## Add a cumsum dat
count_frame['cumsums'] = count_frame['percents'].cumsum()
## Print as a nice table
print(count_frame[['number', 'percents', 'cumsums']])

#plt.hist(dat['count'])
plt.hist(dat['count'], bins = 8)
plt.title('Frequency of number of cups of coffee consumed')
plt.xlabel('Cups of coffee per day')
plt.ylabel('Frequency')
plt.show()
print(skew(dat['count']))

print(np.mean(dat['count']))
print(np.median(dat['count']))

print(sms.DescrStatsW(list(dat['count'])).tconfint_mean())