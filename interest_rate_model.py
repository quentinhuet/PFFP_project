import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class InterestRateModel:
    def __init__(self, initial_rate, volatility, mean_reversion, drift, time_step):
        """
        Initialization of the class.
        
        :param initial_rate: Initial interest rate.
        :param volatility: Input volatility.
        :param mean_reversion: Input mean reversion.
        :param drift: Input Long-term mean.
        :param time_step: Time step for the simulation.
        """
        self.initial_rate = initial_rate
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.drift = drift
        self.time_step = time_step
    
    def simulate_euler_maruyama(self, num_steps, num_paths):
        """
        Simulates paths - Euler-Maruyama scheme. 
        
        :param num_steps: Number of time steps in each path.
        :param num_paths: Number of paths to simulate.
        :return: Simulated paths as a numpy array.
        """
        dt = self.time_step
        paths = np.zeros((num_steps + 1, num_paths))
        paths[0] = self.initial_rate
        
        for t in range(1, num_steps + 1):
            z = np.random.normal(size=num_paths)
            paths[t] = paths[t-1] + self.mean_reversion * (self.drift - paths[t-1]) * dt + self.volatility * np.sqrt(dt) * z
        
        return paths
    
    def simulate_milstein(self, num_steps, num_paths):
        """
        Simulates paths - Milstein scheme.
        
        :param num_steps: Number of time steps in each path.
        :param num_paths: Number of paths to simulate.
        :return: Simulated paths as a numpy array.
        """
        dt = self.time_step
        paths = np.zeros((num_steps + 1, num_paths))
        paths[0] = self.initial_rate
        
        for t in range(1, num_steps + 1):
            z = np.random.normal(size=num_paths)
            dw = np.sqrt(dt) * z
            paths[t] = paths[t-1] + self.mean_reversion * (self.drift - paths[t-1]) * dt + self.volatility * paths[t-1] * dw + 0.5 * self.volatility**2 * paths[t-1] * (dw**2 - dt)
        
        return paths
    

    def plot_paths(self, paths):
        """
        Plots the paths that were simulated.
        
        :param paths: Simulated paths as a numpy array.
        """
        plt.plot(paths)
        plt.xlabel('Time Steps')
        plt.ylabel('Interest Rate')
        plt.show()


    def estimate_asset_price(self, paths, n):
        """
        Estimate the asset price at time step n using Monte Carlo simulation.
        
        :param paths: Simulated interest rate paths, output of simulate_euler_maruyama or simulate_milstei.
        :param n: Time step at which we want to estimated the asset price.
        :return: Estimated asset price.
        """
        T = n * self.time_step  # Time to maturity
        asset_prices = np.exp(-paths[n] * T)
        
        # Average the asset prices across all paths to get the Monte Carlo estimate
        estimated_price = np.mean(asset_prices)
        
        return estimated_price
    
    def print_parameters(self):
        """
        Prints current model parameters.
        """
        print('Model parameters')
        print(f'Volatility : {self.volatility}')
        print(f'Mean Reversion : {self.mean_reversion}')
        print(f'Drift : {self.drift}')

    def calibrate_parameters(self, historical_rates):
        """
        Calibrates model parameters to historical data using Maximum Likelihood Estimation.
        
        :param historical_rates: Array of historical interest rates.
        """
        def negative_log_likelihood(params):
            alpha, beta, sigma = params
            log_likelihood = 0
            for t in range(1, len(historical_rates)):
                dt = self.time_step
                dr = historical_rates[t] - historical_rates[t-1]
                mean = alpha * (beta - historical_rates[t-1]) * dt
                variance = sigma**2 * dt
                log_likelihood += -0.5 * np.log(2 * np.pi * variance) - ((dr - mean)**2) / (2 * variance)
            return -log_likelihood
        
        initial_guess = [self.mean_reversion, self.drift, self.volatility]
        bounds = [(0, None), (None, None), (0, None)]
        
        result = minimize(negative_log_likelihood, initial_guess, bounds=bounds)
        self.mean_reversion, self.drift, self.volatility = result.x

