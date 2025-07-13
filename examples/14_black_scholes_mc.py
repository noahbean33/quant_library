import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.pricing import black_scholes_analytical, black_scholes_monte_carlo

def main():
    """Main function to run the Black-Scholes option pricing example."""
    # 1. Define option parameters
    S0 = 100.0     # Initial stock price
    E = 100.0      # Strike price
    T = 1.0        # Time to maturity in years
    rf = 0.05      # Risk-free rate (annual)
    sigma = 0.2    # Annual volatility
    num_simulations = 250000  # Number of simulations for Monte Carlo

    print("--- Black-Scholes European Call Option Pricing ---")
    print(f"Parameters: S0={S0}, E={E}, T={T}, rf={rf}, sigma={sigma}")

    # 2. Price using the analytical formula
    analytical_price = black_scholes_analytical(
        S0=S0, E=E, T=T, rf=rf, sigma=sigma, option_type='call'
    )
    print(f"\nAnalytical Black-Scholes Price: {analytical_price:.4f}")

    # 3. Price using Monte Carlo simulation
    print(f"Running Monte Carlo simulation with {num_simulations} paths...")
    mc_price = black_scholes_monte_carlo(
        S0=S0, E=E, T=T, rf=rf, sigma=sigma, 
        num_simulations=num_simulations, option_type='call'
    )
    print(f"Monte Carlo Simulation Price:   {mc_price:.4f}")

    # 4. Compare the results
    difference = mc_price - analytical_price
    print(f"Difference (MC - Analytical):   {difference:.4f}")

if __name__ == '__main__':
    main()
