#hackaTUM C0dev1d19. April 2020
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

class SEIR_model:

    def __init__(self,N,S,E,I,R,t,location='None'):
        self.N=np.asarray(N)
        self.S=np.asarray(S)
        self.E=np.asarray(E)
        self.I=np.asarray(I)
        self.R=np.asarray(R)
        self.t=np.asarray(t)
        self.location=location

    def model(self,rho):
        N=self.N
        S = self.S
        E = self.E
        I = self.I
        R = self.R
        t = self.t

        n_meas = len(t) #number of measurements
        dt = t[1:]-t[0:n_meas-1] #delta t of the time-series

        #filter out values with dt eq 0
        S = S[ np.hstack((True,dt>0)) ]
        E = E[np.hstack((True, dt > 0))]
        I = I[np.hstack((True, dt > 0))]
        R = R[np.hstack((True, dt > 0))]
        dt = dt[dt > 0]
        n_meas = len(dt) + 1

        #beta
        A = np.asmatrix(-S[1:]*I[1:],dtype=float).T
        b = np.asmatrix((S[1:]-S[0:n_meas-1])/dt,dtype=float).T
        beta = (np.linalg.lstsq(A,b,rcond=None)[0])[0,0]

        #alpha
        A = np.asmatrix(E[1:],dtype=float).T
        b = np.asmatrix(beta*S[1:]*I[1:]-(E[1:]-E[0:n_meas-1])/dt,dtype=float).T
        alpha = (np.linalg.lstsq(A, b,rcond=None)[0])[0,0]

        #gamma
        A = np.asmatrix(I[1:],dtype=float).T
        b = np.asmatrix(alpha*E[1:]-(I[1:]-I[0:n_meas-1])/dt,dtype=float).T
        gamma = (np.linalg.lstsq(A, b,rcond=None)[0])[0,0]

        self.result = np.stack([beta, alpha, gamma]).T
        return self.result

    def filter_data(self):
        pass

    def visualize_results(self,plot=True):
        self.result = self.result[:,0:2]
        lineObjects = plt.plot(self.t[1:], self.result)
        plt.legend(iter(lineObjects), ('beta','alpha', 'gamma'), title='SEIR parameters:')
        plt.title('SEIR. COVID-19. '+self.location)
        plt.ylabel('Parameters')
        plt.xlabel('Time [days]')
        plt.grid(True, which='both')
        if plot == True:
            plt.show()

class RetrieveData(object):

    def __init__(self,source="JHU",use_api=True):
        self.retrieve_data(source=source,use_api=use_api)

    def retrieve_data(self,source="JHU",use_api=True):
        if source == "fusionbase":
            if use_api == True:
                self.retrieve_fusionbase_api(dates_FB=False)
            else:
                self.retrieve_fusionbase_csv(dates_FB=False)
        if source == "JHU":
            self.retrieve_JHU()


    def retrieve_fusionbase_api(self,dates_FB = False):
        location = "München"
        kreis_nuts = "DE212"

        # url = "https://api.public.fusionbase.io/cases/latest"
        url = "https://api.public.fusionbase.io/cases"

        headers = {
            'X-API-Key': 'd20ca43d-9626-43e4-a304-8ff59feec044'
        }

        response = requests.request("GET", url, headers=headers)
        data = json.loads(response.text.encode('utf8'))

        # -----------------------------------------------------
        # Select a region (example: Munich)
        # -----------------------------------------------------
        df = pd.DataFrame(data)
        df = df[df['location_label'] == location]
        df = df[df['kreis_nuts'] == kreis_nuts]

        # -----------------------------------------------------
        # Save to disk
        # -----------------------------------------------------
        df.to_csv(r'/Users/nestor/Documents/hackatumCovid2020/modelling/data/test_muc.csv', index=False,header=True)
        self.df = df

    def retrieve_fusionbase_csv(self,dates_FB = False):
        filename = "/Users/nestor/Documents/hackatumCovid2020/modelling/data/test_muc.csv"
        data = pd.read_csv(filename)
        df = pd.DataFrame(data)
        self.df = df

    def retrieve_JHU(self, country='Germany'):

        self._dir = Path.cwd().parent.joinpath('data/JHU')
        base_file_name = 'time_series_covid19_{}_global.csv'
        file_types = ['confirmed', 'deaths', 'recovered']
        file_list = [self._dir.joinpath(base_file_name.format(f)) for f in file_types]

        country_filter = lambda df, ctr: df.loc[df['Country/Region'] == ctr]

        raw_data = [pd.read_csv(file) for file in file_list]
        [I, D, R] = [country_filter(data, country).T for data in raw_data]
        I.index = pd.to_datetime(I.index, errors='coerce', infer_datetime_format=True)

        data_dict = dict([(type_, df.iloc[4:,0]) for df, type_ in zip([I, D, R], file_types)])
        self.__setattr__(country, pd.DataFrame(data_dict, columns= file_types))


    def get_data(self, country= None):
        if country is None:
            return self.df
        elif hasattr(self, country):
            return self.__getattribute__(country)
        else:
            raise AttributeError('RertriveData instance does not have {} data present'.format(country))

if __name__ == '__main__':
    data_ret = RetrieveData(source="JHU")
    '''
    dates_FB = False
    location = "München"
    kreis_nuts = "DE212"
    if (dates_FB == False):
        FMT = '%Y-%m-%dT%H:%M:%S'
        date_0 = datetime.strptime("2020-01-01T00:00:00", FMT)
    else:
        FMT = '%Y-%m-%dT%H:%M:%S.%f'
        date_0 = datetime.strptime("2020-01-01T00:00:00.0", FMT)
    '''
    #data_ret.retrieve_data()
    df = data_ret.get_data(country= 'Germany')
    '''
    #dataret = retrieve_data(source = "JHU")
    #df = dataret.get_data()
    
    list(df.columns)
    #overall_cases = df['cases'].sum()
    #max_cases = df.loc[df['cases'].idxmax()]
    #print(max_cases)
    
    #Obtaining time-series times
    if (dates_FB==False):
        dates = df['publication_datetime']
    else:
        dates = df['fb_datetime']
    junk = dates.map(lambda x: (datetime.strptime(x, FMT) - date_0).days)
    df.insert(13, "Days", junk, True)

    #Sorting data according to days
    #muc_df["Days"].sort_values
    df.sort_values('Days',ascending=True,inplace=True)

    #Remove sparse data from Munich dataset
    df = df.iloc[2:, ]

    #Assume no social distance. That means that S=E
    N = (df['population'].iloc[0])
    '''
    N = 80e6
    I = df['confirmed']
    # R = I * 1/10  # This should be an input data
    R = df['recovered'] + df['deaths']
    S = N - I - R
    E = S * 0.7  #let's assume that the 70% of the total sussceptible population is exposed
    t = [t_ for t_ in range(len(df.index))]
    seir = SEIR_model(N,S,E,I,R,t)

    beta, alpha, gamma = seir.model(rho=1)  # no social distancing
    print("beta = " + str(beta))
    print("alpha = " + str(alpha))
    print("gamma = " + str(gamma))


    #seir.visualize_results(plot=True)
