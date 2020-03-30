import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np 
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.stats import linregress


class Predictor: 

    def __init__(self, data_group, model): 
        self.data = data_group
        self.model = model
        self.predictor = None

    def fit_data(self, value_type = 'positive_total'): 
        
        group_infected = self.data.groupby('value_type').get_group(value_type)
        
        dates = group_infected['time_report'].unique()
        datetime_dates = pd.to_datetime(dates)

        self.start_time = min(datetime_dates)
        self.end_time = max(datetime_dates)
        self.value_type = value_type

        times = np.array([])
        case_numbers = np.array([])

        for date in dates: 
            values_date = group_infected.loc[group_infected['time_report']== date]['value']
            values_date = self.clean_value_series(values_date)
            case_numbers = np.append(case_numbers, values_date.sum()) 
            
            delta = (pd.to_datetime(date) - self.start_time).days
            times = np.append(times, int(delta))

        self.fit_range = int( (max(datetime_dates) - min(datetime_dates)).days )

        init_values = self.model.get_init_values(times, case_numbers)
        self.predictor = self.fit_data_to_model(times, case_numbers, self.model.function, init_values)
        print(self.predictor)

    def extrapolate(self, timespan, filepath = None): 
        if self.predictor is None:
            raise Exception('no fitted model available!')
        days = [d for d in range(timespan)]

        self.extrapolation_timespan = timespan
        datetimes, values = self.get_extrapolated_values(days)
        self.data_extrapolated = self.append_new_data(values, datetimes) 
        if not filepath is None: 
            self.write_csv(self.data_extrapolated, filepath)
            return

        return self.data_extrapolated

    def fit_data_to_model(self, times, values, model, init_values):
        """
        return extrapolation model
        out: fun, function handle for extrapolation
        """
        fit = curve_fit(self.model.function, times, values, p0=init_values)
        fit_parameters = fit[0]

        errors = [np.sqrt(fit[1][i][i]) for i in [0,1]]
        print('fitting errors: ', errors)
        fun_handle = lambda t: self.model.function(t, *fit_parameters)

        return fun_handle, errors

    def get_extrapolated_values(self, times_new):
        """
        Return the extrapolated values from 
        """
        extrapolated_values = self.predictor(times_new)
        time_deltas = [ timedelta(days = d) for d in times_new ]
        time_stamps_new = [ self.end_time + delta for delta in time_deltas ]

        return time_stamps_new, extrapolated_values

    def append_new_data(self, new_values, new_timestamps): 

        data_copy = self.data.groupby('value_type').get_group(self.value_type)
        data_copy = data_copy[ -self.extrapolation_timespan: ]
        data_copy['values'] = new_values
        data_copy['time_report'] = new_timestamps  
        data_extrapolated = pd.concat([self.data, data_copy], ignore_index= True)     
        
        return data_extrapolated

    def clean_value_series(self, values): 
        for ind, val in enumerate(values): 
            try: 
                val = int(val)
            except: 
                val = 0
            values.iloc[ind] = val
        return values

class LogisticRegression: 

    def __init__(self):
        pass 

    def function(self, x, b, a):

        return 1./(1. + a * np.exp(-(b*x)))

    def get_init_values(self, x, y): 

        y_tr = np.log(1./y - 1)
        res = linregress(x,y)

        return res[0], np.exp(res[1])



if __name__=='main': 
    
    filepath = 'file\path\goes\here'
    filename = 'uk_harmonized.csv'
    data = pd.read_csv(filepath)
    model = LogisticRegression()
    prediction_span = 10       # number of days to look into the future

    for name_region, data_region in data.groupby('region_name'): 
    
        if name_region in ['Scotland', 'England', 'Wales', 'Northerm Ireland']: 
            
            predictor_region = Predictor(data_region, model)
            predictor_region.fit_data(value_type = 'positive_total')
            data_region_extrap = predictor_region.extrapolate(prediction_span)
            data = pd.concat([data, data_region_extrap])
    
    data.to_csv(os.path.join(filepath,str(prediction_span)+ 'day_extra'+ filename))

            


       

