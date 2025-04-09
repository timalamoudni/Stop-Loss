import numpy as np
import matplotlib.pyplot as plt



class GBMSimulator:
    def __init__(self, S0, r, sigma, T, N):

        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N

    def simulate_path(self):
        """Simule une trajectoire de l’actif selon la discrétisation d’Euler (équation A.5)."""
        S = np.zeros(self.N + 1)
        S[0] = self.S0
        for t in range(1, self.N + 1):
            z = np.random.normal()  
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * z * np.sqrt(self.dt))
        return S

    def plot_path(self, S_path):
        """Affiche la trajectoire simulée."""
        times = np.linspace(0, self.T, self.N + 1)
        plt.figure()
        plt.plot(times, S_path, lw=2)
        plt.xlabel("Temps")
        plt.ylabel("Prix")
        plt.title("Trajectoire simulée de l'actif")
        plt.grid(True)
        plt.show()