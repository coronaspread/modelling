import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np 
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.stats import linregress



def logistic_model(x, b, a):
    return 1./(1. + a * np.exp(-(b*x)))

def get_init_vals_logistic(x,y): 
    
    y_tr = np.log(1./y - 1)
    res = linregress(x,y)

    return res[0], np.exp(res[1])

class DatetimeConverter: 

    def __init__(self, time_stamps): 
        self._initial = min(time_stamps)
        self._range = int((max(time_stamps) - min(time_stamps)).days)

    def get_date_range(self): 
        return self._range

    def get_initial_date(self): 
        return self._initial

    def get_date(day_numbers):
        
        if not isinstance(day_numbers, list): 
            day_numbers = [day_numbers]
        
        dates = [] 
        for day in day_numbers: 
            dates.append(self._initial + timedelta(days = day))

        if len(dates) == 1:
            return dates[0]
        
        return dates 
 
        
def fit_data_to_model(self,times, values, model, init_values):
    """
    return extrapolation model
    out: fun, function handle for extrapolation
    """
    fit = curve_fit(model, times, values, p0=init_values)
    fit_parameters = fit[0]

    errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
    fun_handle = lambda t: model(t, *fit_parameters)

    return fun_handle, errors

def get_extrapolated_values(trained_model, times_new):
    extrapolated_values = trained_model(times_new)
    time_deltas = [ timedelta(days = d) for d in times_new ]
    time_stamps_new = [ start_time + delta for delta in time_deltas ]

    return time_stamps_new, extrapolated_values

"""
Main loop
"""
for name_region, data_region in data.groupby('region_name'): 
    
    if name_region in ['Scotland', 'England', 'Wales', 'Northerm Ireland']: 
        
        predictor_region = fit_region(data_region)

        group_infected = region_data.groupby('value_type').get_group('positive_total')

        dates = pd.to_datetime(group_infected['time_report'])

        date_converter =  DatetimeConverter(dates)
        fit_range = date_converter.get_date_range()
        start_time = min(dates)

        times = []
        case_numbers = []

        for date in dates: 
            delta = (date - start_time).days
            times.append(int(times))

        case_numbers = group_infected['value'].iloc[:]
        
        init_values = get_init_vals_logistic(times, case_numbers)
        predictor = fit_data_to_model(times, case_numbers, logistic_model, init_values)

        get_extrapolated_values(predictor, times_extrapolate)


    

