import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.simulation.stochastic_processes import simulate_ornstein_uhlenbeck
from src.valueinvestpy.visuals.plotting import plot_stochastic_process as plot_process

def main():
    """Main function to run the Ornstein-Uhlenbeck process simulation example."""
    # 1. Define simulation parameters for a mean-reverting process (e.g., interest rate)
    initial_value = 0.03    # Starting value (e.g., 3% interest rate)
    reversion_speed = 0.5   # Speed of reversion to the mean
    long_term_mean = 0.05   # Long-term mean (e.g., 5% interest rate)
    volatility = 0.02       # Annual volatility (2%)
    time_horizon = 2        # Simulation time in years
    num_simulations = 50    # Number of paths to simulate

    print("--- Ornstein-Uhlenbeck Process Simulation ---")
    print(f"Initial Value: {initial_value:.2%}")
    print(f"Long-Term Mean (mu): {long_term_mean:.2%}")
    print(f"Reversion Speed (theta): {reversion_speed}")
    print(f"Annual Volatility (sigma): {volatility:.2%}")
    print(f"Simulating {num_simulations} paths over {time_horizon} year(s)...")

    # 2. Run the Ornstein-Uhlenbeck simulation
    simulation_df = simulate_ornstein_uhlenbeck(
        x0=initial_value,
        theta=reversion_speed,
        mu=long_term_mean,
        sigma=volatility,
        T=time_horizon,
        num_simulations=num_simulations
    )

    # 3. Print the last day's stats
    last_day_values = simulation_df.iloc[-1]
    print("\n--- Simulation Results (at end of period) ---")
    print(f"Average Final Value: {last_day_values.mean():.2%}")
    print(f"Median Final Value: {last_day_values.median():.2%}")

    # 4. Plot the simulation results
    print("\nDisplaying simulation plot...")
    plot_process(
        simulation_df, 
        title=f'Ornstein-Uhlenbeck Process - {num_simulations} Simulations',
        xlabel='Time (Years)',
        ylabel='Value (e.g., Interest Rate)'
    )

if __name__ == '__main__':
    main()
