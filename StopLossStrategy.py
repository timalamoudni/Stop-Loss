import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from GBS import GBMSimulator

class StopLossStrategy:
    def __init__(self, S0, x, r_f, T, N, sigma, r=None, transaction_fee=0.0):
        """
        S0              : valeur initiale de l'actif risqué
        x               : pourcentage de garantie souhaité (ex. 0.9 pour garantir 90% de S0)
        r_f             : taux de l'actif sans risque
        T               : horizon 
        N               : nombre d'intervalles
        sigma           : volatilité de l'actif risqué
        r               : taux de l'actif risqué 
        transaction_fee : frais de transaction 
        """
        self.S0 = S0
        self.x = x
        self.r_f = r_f
        self.T = T
        self.N = N
        self.sigma = sigma
        self.dt = T / N
        self.r = r if r is not None else r_f
        self.transaction_fee = transaction_fee
        
        
        self.M0 = self.x * self.S0 * np.exp(-self.r_f * self.T)
    
    def simulate(self, rebalance_freq=1, tol=0.0):
        """
        Simule l'évolution d'un portefeuille géré selon la stratégie Stop Loss.
        
        Paramètres :
          rebalance_freq : fréquence de recomposition(2 pour chaque 2 pas)
          tol            : toléranc
        """
        times = np.linspace(0, self.T, self.N + 1)
        
        gbm = GBMSimulator(self.S0, self.r, self.sigma, self.T, self.N)
        S_path = gbm.simulate_path()
        
        port_values = np.zeros(self.N + 1)
        port_values[0] = self.S0  
        
        
        state = "risky"
        
        shares = 1.0  
        cash = 0.0

        for i in range(1, self.N + 1):
            t = times[i]
            
            M_t = self.M0 * np.exp(self.r_f * t)
            
            if i % rebalance_freq == 0:
                if state == "risky":
                    
                    V = shares * S_path[i]
                    
                    if S_path[i] < (1 - tol) * M_t:
                        
                        fee = self.transaction_fee * V
                        cash = V - fee
                        shares = 0.0
                        state = "risk_free"
                        V = cash
                    port_values[i] = V
                else:  
                    
                    cash = cash * np.exp(self.r_f * self.dt)
                    
                    if S_path[i] > (1 + tol) * M_t:
                        
                        fee = self.transaction_fee * cash
                        available_cash = cash - fee
                        shares = available_cash / S_path[i]
                        cash = 0.0
                        state = "risky"
                        V = shares * S_path[i]
                    else:
                        V = cash
                    port_values[i] = V
            else:
                
                if state == "risky":
                    V = shares * S_path[i]
                else:
                    
                    cash = cash * np.exp(self.r_f * self.dt)
                    V = cash
                port_values[i] = V
        
        return times, S_path, port_values

    def plot_results(self, times, S_path, port_values):
        
        M_t = self.M0 * np.exp(self.r_f * times)
        plt.figure()
        plt.plot(times, S_path, label="Prix de l'actif risqué", lw=2)
        plt.plot(times, port_values, label="Valeur du portefeuille", lw=2)
        plt.plot(times, M_t, label="Plancher garanti", lw=2, linestyle='--')
        plt.xlabel("Temps")
        plt.ylabel("Valeur")
        plt.title("Simulation de la stratégie Stop Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    S0 = 100         
    r = 0.05         
    sigma = 0.2      
    T = 1            
    N = 252          
    x = 0.9          
    r_f = 0.02       
    fee = 0.005     
    
    strategy = StopLossStrategy(S0, x, r_f, T, N, sigma, r=r, transaction_fee=fee)
    times, S_path, port_values = strategy.simulate(rebalance_freq=1, tol=0)
    strategy.plot_results(times, S_path, port_values)