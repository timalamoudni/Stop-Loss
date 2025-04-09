from SensitivityAnalysis import * 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, gaussian_kde


def plot_michelin_density(csv_file):
    """
    Lit les cours de l’action Michelin depuis le fichier CSV,
    calcule les taux de rentabilité (ici en log-return),
    et trace sur un même graphique l'estimation de densité (KDE) et la densité gaussienne.
    """
    df = pd.read_csv(csv_file, sep=';')
    df.sort_values(by="Date", inplace=True)
    prix = df["Closing"].values
    prix = [float(i.replace(",",".")) for i in prix]
    log_returns = np.diff(np.log(prix))
    log_returns.sort()
    
    kde = gaussian_kde(log_returns)
    x_vals = np.linspace(log_returns.min(), log_returns.max(), 1000)
    kde_vals = kde(x_vals)
    
    mu, sigma_est = np.mean(log_returns), np.std(log_returns)
    gaussian_vals = norm.pdf(x_vals, loc=mu, scale=sigma_est)
    
    plt.figure()
    plt.plot(x_vals, kde_vals, label="Densité de Michelin", lw=2)
    plt.plot(x_vals, gaussian_vals, label="Densité gaussienne", lw=2, linestyle="--")
    plt.xlabel("Taux de rentabilité")
    plt.ylabel("Densité")
    plt.title("Comparaison des densités : Michelin vs Gaussienne")
    plt.legend()
    plt.grid(True)
    plt.show()
    
S0 = 100         
r = 0.05         
sigma = 0.2      
T = 1            
N = 252          


x = 0.9          
r_f = 0.02       
fee = 0.005 
base_params = {
    
    "S0": S0,
    "x": x,
    "r_f": r_f,
    "T": T,
    "N": N,
    "sigma": sigma,
    "r": r,
    "tol": 0.05,
    "rebalance_freq": 1,
    'transaction_fee':0
}

print("Affichage de la densité de Michelin")
plot_michelin_density("Michelin_20112023_20112024.csv")


sensitivity = SensitivityAnalysis(**base_params)

print("Affichage de l'analyse de sigma'")
sigma_values = [0.1, 0.2, 0.3, 0.4]
sensitivity.analyze_parameter('sigma', sigma_values, num_simulations=1000)

print("Affichage de l'analyse du mantant a garantir'")
x_values = [0.8, 0.9, 1.0]
sensitivity.analyze_parameter('x', x_values, num_simulations=1000)

print("Affichage de l'analyse de la tolerance'")
tol_values = [0.0, 0.05, 0.1]
sensitivity.analyze_parameter('tol', tol_values, num_simulations=1000)

print("Affichage de l'analyse de la rebalance freq'")
freq_values = [1, 5, 10]
sensitivity.analyze_parameter('rebalance_freq', freq_values, num_simulations=1000)

print("Affichage de l'analyse des frais'")
fee_values = [0, 0.005, 0.01]
sensitivity.analyze_parameter('transaction_fee', fee_values, num_simulations=1000)