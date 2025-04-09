import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, gaussian_kde

from StopLossStrategy import StopLossStrategy



class SensitivityAnalysis:
    def __init__(self, S0, x, r_f, T, N, sigma, r=None, tol=0.0, rebalance_freq=1, transaction_fee=0):

        self.S0 = S0
        self.x = x
        self.r_f = r_f
        self.T = T
        self.N = N
        self.sigma = sigma
        self.r = r if r is not None else r_f
        self.tol = tol
        self.rebalance_freq = rebalance_freq
        self.transaction_fee = transaction_fee

    def simulate_strategy(self, x=None, sigma=None, tol=None, rebalance_freq=None, transaction_fee=None, num_simulations=1000):
 
        if x is None:
            x = self.x
        if sigma is None:
            sigma = self.sigma
        if tol is None:
            tol = self.tol
        if rebalance_freq is None:
            rebalance_freq = self.rebalance_freq
        if transaction_fee is None:
            transaction_fee = self.transaction_fee

        terminal_values = []
        for _ in range(num_simulations):
            strat = StopLossStrategy(self.S0, x, self.r_f, self.T, self.N, sigma, self.r, transaction_fee=transaction_fee)
            _, _, port_vals = strat.simulate(rebalance_freq=rebalance_freq, tol=tol)
            terminal_values.append(port_vals[-1])
        return np.array(terminal_values)

    def analyze_parameter(self, param_name, param_values, num_simulations=1000):

        results = {}
        plt.figure(figsize=(8, 6))
        for val in param_values:
            if param_name == 'x':
                term_vals = self.simulate_strategy(x=val, num_simulations=num_simulations)
            elif param_name == 'sigma':
                term_vals = self.simulate_strategy(sigma=val, num_simulations=num_simulations)
            elif param_name == 'tol':
                term_vals = self.simulate_strategy(tol=val, num_simulations=num_simulations)
            elif param_name == 'rebalance_freq':
                term_vals = self.simulate_strategy(rebalance_freq=val, num_simulations=num_simulations)
            elif param_name == 'transaction_fee':
                term_vals = self.simulate_strategy(transaction_fee=val, num_simulations=num_simulations)
            else:
                raise ValueError("Paramètre non reconnu.")
            results[val] = term_vals
            kde = gaussian_kde(term_vals)
            x_vals = np.linspace(term_vals.min(), term_vals.max(), 1000)
            plt.plot(x_vals, kde(x_vals), lw=2, label=f"{param_name} = {val}")
        plt.title(f"Sensibilité de la valeur terminale en fonction de {param_name}")
        plt.xlabel("Valeur terminale du portefeuille")
        plt.ylabel("Densité estimée")
        plt.legend()
        plt.grid(True)
        plt.show()
        return results