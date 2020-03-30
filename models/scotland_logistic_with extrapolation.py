# -*- coding: utf-8 -*-
# derived from https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# %matplotlib inline

TIME_0 = "2020-01-01"
FMT = '%Y-%m-%d'
date_0 = datetime.strptime("2020-01-01", FMT)

# get data
url = "https://raw.githubusercontent.com/codevscovid19/modelling/master/data/2020-03-29_scotland_testing.csv"
df = pd.read_csv(url)

# prepare data
df = df.loc[:,['time_report','value']]
date = df['time_report']
df['time_report'] = date.map(lambda x : (datetime.strptime(x, FMT) - date_0).days  )

# logistic model
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

# fitting the curve
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])

fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])

# determine errors
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]

# results of logistic funtion
a = fit[0][0]
b = fit[0][1]
c = fit[0][2]
print("Expected number of infected people: {0} +/- {1}".format(c, errors[2]))
peak_date = date_0 + timedelta(days=int(b))
print("Expected infection peak: {0}".format(peak_date))

# estimate end of infection
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
end_date = date_0 + timedelta(days=int(sol))
print("Expected infection end: {0}".format(end_date))

# exponential model
def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

# fit exponential curve
try:
    exp_fit = curve_fit(exponential_model,x,y,p0=[1,1,1])
except:
    print("no exponential fit found in 800 iterations")
    exp_fit = None

# plot
pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]

plt.rc('font', size=14)

# Real data
plt.scatter(x,y,label="Real data",color="red")

# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2])
    for i in x+pred_x], label="Logistic model" )

# Predicted exponential curve
if exp_fit:
    plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2])
        for i in x+pred_x], label="Exponential model" )

plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

plt.show()

# extrapolation
infected_extrapolation = {}
for i in range(max(x), max(x) + 20):
    infected[i] = logistic_model(i, a, b, c)

print(infected_extrapolation)
