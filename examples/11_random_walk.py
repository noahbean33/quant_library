import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.simulation.stochastic_processes import simulate_random_walk
from src.valueinvestpy.visuals.plotting import plot_stochastic_process as plot_process

def main():
    """Main function to run the random walk simulation example."""
    # 1. Define simulation parameters
    num_steps = 1000
    num_simulations = 50

    print("--- Simple Random Walk Simulation ---")
    print(f"Simulating {num_simulations} paths with {num_steps} steps each...")

    # 2. Run the random walk simulation
    simulation_df = simulate_random_walk(
        num_steps=num_steps,
        num_simulations=num_simulations
    )

    # 3. Print the final position's stats
    final_positions = simulation_df.iloc[-1]
    print("\n--- Simulation Results (at end of walk) ---")
    print(f"Average Final Position: {final_positions.mean():.4f}")
    print(f"Standard Deviation of Final Position: {final_positions.std():.4f}")

    # 4. Plot the simulation results
    print("\nDisplaying simulation plot...")
    plot_process(
        simulation_df, 
        title=f'Simple Random Walk - {num_simulations} Simulations',
        xlabel='Steps',
        ylabel='Position'
    )

if __name__ == '__main__':
    main()
