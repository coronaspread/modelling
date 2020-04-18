#hackaTUM C0dev1d19. April 2020
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


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

        #beta
        beta = (S[0:n_meas-1]-S[1:]) / S[1:]*I[1:]*dt
        print(beta)

        #alpha
        alpha = 1/E[1:] * (beta*S[1:]*I[1:]-(E[1:]-E[0:n_meas-1])/dt)

        #gamma
        gamma = 1/I[1:]*(alpha*E[1:]-(I[1:]-I[0:n_meas-1])/dt)

        self.result = np.stack([beta, alpha, gamma]).T

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

class retrieve_data:

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

    def retrieve_JHU(self):
        filename = "/Users/nestor/Documents/hackatumCovid2020/modelling/data/"


    def get_data(self):
        return self.df


if __name__ == '__main__':

    dates_FB = False
    location = "München"
    kreis_nuts = "DE212"
    if (dates_FB == False):
        FMT = '%Y-%m-%dT%H:%M:%S'
        date_0 = datetime.strptime("2020-01-01T00:00:00", FMT)
    else:
        FMT = '%Y-%m-%dT%H:%M:%S.%f'
        date_0 = datetime.strptime("2020-01-01T00:00:00.0", FMT)



    dataret = retrieve_data(source = "fusionbase",use_api=False)
    df = dataret.get_data()

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

    I = df['cases']
    R = I * 1/10  # This should be an input data
    S = N - I - R
    E = S * 0.7  #let's assume that the 70% of the total sussceptible population is exposed

    seir = SEIR_model(N,S,E,I,R,df["Days"],location=location)
    seir.model(rho=1)   #no social distancing
    seir.visualize_results(plot=True)




