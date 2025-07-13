import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.simulation.stochastic_processes import simulate_gbm
from src.valueinvestpy.visuals.plotting import plot_stochastic_process as plot_process

def main():
    """Main function to run the Geometric Brownian Motion (GBM) simulation example."""
    # 1. Define simulation parameters
    initial_price = 100  # Starting stock price
    drift = 0.05         # Expected annual return (5%)
    volatility = 0.2     # Annual volatility (20%)
    time_horizon = 1     # Simulation time in years
    num_simulations = 100 # Number of paths to simulate

    print("--- Geometric Brownian Motion Simulation ---")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Annual Drift (mu): {drift:.2%}")
    print(f"Annual Volatility (sigma): {volatility:.2%}")
    print(f"Simulating {num_simulations} paths over {time_horizon} year(s)...")

    # 2. Run the GBM simulation
    simulation_df = simulate_gbm(
        S0=initial_price,
        mu=drift,
        sigma=volatility,
        T=time_horizon,
        num_simulations=num_simulations
    )

    # 3. Print the last day's price stats
    last_day_prices = simulation_df.iloc[-1]
    print("\n--- Simulation Results (at end of year) ---")
    print(f"Average Price: ${last_day_prices.mean():.2f}")
    print(f"Median Price: ${last_day_prices.median():.2f}")
    print(f"Max Price: ${last_day_prices.max():.2f}")
    print(f"Min Price: ${last_day_prices.min():.2f}")

    # 4. Plot the simulation results
    print("\nDisplaying simulation plot...")
    plot_process(
        simulation_df, 
        title=f'Geometric Brownian Motion - {num_simulations} Simulations',
        xlabel='Time (Years)',
        ylabel='Stock Price'
    )

if __name__ == '__main__':
    main()
