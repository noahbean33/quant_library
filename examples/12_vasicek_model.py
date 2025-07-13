import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.simulation.stochastic_processes import simulate_vasicek_model
from src.valueinvestpy.visuals.plotting import plot_stochastic_process as plot_process

def main():
    """Main function to run the Vasicek model simulation example."""
    # 1. Define Vasicek model parameters
    r0 = 0.03       # Initial interest rate (3%)
    kappa = 0.15    # Speed of reversion
    theta = 0.05    # Long-term mean interest rate (5%)
    sigma = 0.01    # Volatility (1%)
    T = 10          # Time horizon in years
    N = 252 * T     # Daily steps
    num_simulations = 10

    print("--- Vasicek Interest Rate Model Simulation ---")
    print(f"Simulating {num_simulations} paths over {T} year(s)...")
    print(f"Parameters: r0={r0}, kappa={kappa}, theta={theta}, sigma={sigma}")

    # 2. Run the Vasicek model simulation
    simulation_df = simulate_vasicek_model(
        r0=r0, kappa=kappa, theta=theta, sigma=sigma, 
        T=T, N=N, num_simulations=num_simulations
    )

    # 3. Print the final interest rate stats
    final_rates = simulation_df.iloc[-1]
    print("\n--- Simulation Results (at end of period) ---")
    print(f"Average Final Interest Rate: {final_rates.mean():.4f}")
    print(f"Standard Deviation of Final Rate: {final_rates.std():.4f}")

    # 4. Plot the simulation results
    print("\nDisplaying simulation plot...")
    plot_process(
        simulation_df, 
        title=f'Vasicek Model - {num_simulations} Simulations',
        xlabel='Time (Years)',
        ylabel='Interest Rate'
    )

if __name__ == '__main__':
    main()
