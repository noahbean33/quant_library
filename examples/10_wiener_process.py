import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.simulation.stochastic_processes import simulate_wiener_process
from src.valueinvestpy.visuals.plotting import plot_stochastic_process as plot_process

def main():
    """Main function to run the Wiener process simulation example."""
    # 1. Define simulation parameters
    time_horizon = 2        # Simulation time in years
    num_simulations = 50    # Number of paths to simulate
    time_steps = 504        # Number of steps (e.g., 252 * 2 years)

    print("--- Wiener Process (Brownian Motion) Simulation ---")
    print(f"Simulating {num_simulations} paths over {time_horizon} year(s)...")

    # 2. Run the Wiener process simulation
    simulation_df = simulate_wiener_process(
        T=time_horizon,
        N=time_steps,
        num_simulations=num_simulations
    )

    # 3. Print the last day's stats
    last_day_values = simulation_df.iloc[-1]
    print("\n--- Simulation Results (at end of period) ---")
    print(f"Average Final Value: {last_day_values.mean():.4f}")
    print(f"Standard Deviation of Final Value: {last_day_values.std():.4f}")

    # 4. Plot the simulation results
    print("\nDisplaying simulation plot...")
    plot_process(
        simulation_df, 
        title=f'Wiener Process - {num_simulations} Simulations',
        xlabel='Time (Years)',
        ylabel='W(t)'
    )

if __name__ == '__main__':
    main()
