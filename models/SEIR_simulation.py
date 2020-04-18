#hackaTUM C0dev1d19. April 2020
import numpy as np
import matplotlib.pyplot as plt


class SEIR_simu:

    def __init__(self,init_vals, params, t):
        self.init_vals = init_vals
        S_0, E_0, I_0, R_0 = self.init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        self.params = params
        self.t = t
        self.dt = t[1] - t[0]

    def model(self,rho):
        S_0, E_0, I_0, R_0 = self.init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        alpha, beta, gamma = self.params

        for _ in self.t[1:]:
            next_S = S[-1] - (rho*beta * S[-1] * I[-1]) * self.dt
            next_E = E[-1] + (rho*beta * S[-1] * I[-1] - alpha * E[-1]) * self.dt
            next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * self.dt
            next_R = R[-1] + (gamma * I[-1]) * self.dt
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
        self.result = np.stack([S, E, I, R]).T

    def visualize_results(self,plot=True):
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(2, 1, 1)
        results_SE = self.result[:,1:3]
        lineObjects = plt.plot(self.t, results_SE)
        plt.legend(iter(lineObjects), ('Exposed', 'Infected'), title='Compartment:')
        alpha, beta, gamma = self.params
        plt.title('Baseline COVID-19 SEIR Model (alpha ='+ str(alpha)+', beta ='+str(beta)+', gamma ='+str(gamma)+')')
        plt.ylabel('Population fraction')
        plt.xlabel('Time [days]')
        plt.grid(True, which='both')
        #plt.show()

        ax1 = fig.add_subplot(2, 1, 2)
        lineObjects = plt.plot(self.t, self.result)
        plt.legend(iter(lineObjects), ('Susceptible','Exposed', 'Infected', 'Recovered'), title='Compartment:')
        plt.title('Baseline COVID-19 SEIR Model (alpha ='+ str(alpha)+', beta ='+str(beta)+', gamma ='+str(gamma)+')')
        plt.ylabel('Population fraction')
        plt.xlabel('Time [days]')
        plt.grid(True, which='both')
        if plot == True:
            plt.show()

if __name__ == '__main__':

    # Define parameters
    t_max = 100
    dt = .1
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    N = 10000
    init_vals = 1 - 1 / N, 1 / N, 0, 0
    alpha = 0.2
    beta = 1.75
    gamma = 0.5
    params = alpha, beta, gamma
    # Run simulation
    sim = SEIR_simu(init_vals,params,t)

    rho = 1 # no social distancing
    sim.model(rho)
    sim.visualize_results(plot=False)

    rho = 0.5  # with social distancing
    sim.model(rho)
    sim.visualize_results(plot=True)