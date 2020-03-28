import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
#import matplotlib.pyplot as plt
# %matplotlib inline

TIME_0 = "2020-01-01T00:00:00"
FMT = '%Y-%m-%dT%H:%M:%S'
date_0 = datetime.strptime("2020-01-01T00:00:00", FMT)

# get data
url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)

# prepare data
df = df.loc[:,['data','totale_casi']]
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - date_0).days  )

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
