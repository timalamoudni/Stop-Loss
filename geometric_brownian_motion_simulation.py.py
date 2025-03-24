import numpy as np
import matplotlib.pyplot as plt

class GeometricBrownianMotionSimulator:
    def __init__(self, S0, r, sigma, T, N):
        """
        Initialise le simulateur.
        
        Paramètres :
        - S0 : Prix initial de l'actif
        - r : Taux d'intérêt sans risque
        - sigma : Volatilité de l'actif
        - T : Durée totale de la simulation (en années)
        - N : Nombre de pas de discrétisation (par exemple, nombre de jours de trading)
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N  # pas de temps

    def simulate_path(self):
        """
        Simule une trajectoire de prix selon la formule :
        S(t + dt) = S(t) * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        où Z est un nombre aléatoire issu d'une loi normale standard.
        """
        S = np.zeros(self.N + 1)
        S[0] = self.S0
        # Générer les variables aléatoires pour chaque pas
        Z = np.random.normal(0, 1, self.N)
        for t in range(1, self.N + 1):
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z[t-1])
        return S

    def plot_path(self, S):
        """
        Affiche la trajectoire simulée.
        """
        time_grid = np.linspace(0, self.T, self.N + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, S, lw=2)
        plt.xlabel("Temps")
        plt.ylabel("Prix de l'actif")
        plt.title("Simulation du Mouvement Brownien Géométrique")
        plt.grid(True)
        plt.show()

# Exemple d'utilisation
if __name__ == '__main__':
    # Paramètres de simulation
    S0 = 100         # Prix initial
    r = 0.05         # Taux sans risque (5%)
    sigma = 0.2      # Volatilité (20%)
    T = 1            # Durée de 1 an
    N = 252          # Nombre de pas (par exemple, 252 jours de trading)

    # Création de l'instance du simulateur
    simulator = GeometricBrownianMotionSimulator(S0, r, sigma, T, N)
    # Simulation de la trajectoire
    S_path = simulator.simulate_path()
    # Affichage du résultat
    simulator.plot_path(S_path)
