from statsmodels.stats.power import tt_ind_solve_power
a = tt_ind_solve_power(effect_size=0.1, nobs1 = None, alpha=0.05, power=0.9, ratio=1, alternative='two-sided')
b = tt_ind_solve_power(effect_size=0.25, nobs1 = None, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
c = tt_ind_solve_power(effect_size=None, nobs1 = 200, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
print (a,b,c)
d_values = [x/100.0 for x in range(5,55,5)] # Need range to 55 since Python is zero based indexing

powers = [tt_ind_solve_power(effect_size=d, nobs1 = 200, alpha=0.05, power=None, ratio=1, alternative='two-sided')
            for d in d_values]
print(powers)

d = tt_ind_solve_power(effect_size=None, nobs1 = 20, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
e = tt_ind_solve_power(effect_size=.25, nobs1 = 20, alpha=0.05, power=None, ratio=1, alternative='two-sided')
print(d, e)