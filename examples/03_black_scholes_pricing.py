import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.financial_models import calculate_black_scholes

def main():
    """Main function to run the Black-Scholes option pricing example."""
    # 1. Define the parameters for the option
    underlying_price = 100  # Current price of the stock
    strike_price = 100      # The price at which the option can be exercised
    time_to_expiry = 1      # Time to expiration in years
    risk_free_rate = 0.05   # Annual risk-free interest rate
    volatility = 0.2        # Volatility of the stock's returns (20%)

    print("--- Black-Scholes Option Pricing Model ---")
    print(f"Underlying Price: ${underlying_price:.2f}")
    print(f"Strike Price: ${strike_price:.2f}")
    print(f"Time to Expiry: {time_to_expiry} year(s)")
    print(f"Risk-Free Rate: {risk_free_rate:.2%}")
    print(f"Volatility: {volatility:.2%}")

    # 2. Calculate the price of a European call option
    call_price = calculate_black_scholes(
        S=underlying_price,
        E=strike_price,
        T=time_to_expiry,
        rf=risk_free_rate,
        sigma=volatility,
        option_type='call'
    )
    print(f"\nCalculated Call Option Price: ${call_price:.2f}")

    # 3. Calculate the price of a European put option
    put_price = calculate_black_scholes(
        S=underlying_price,
        E=strike_price,
        T=time_to_expiry,
        rf=risk_free_rate,
        sigma=volatility,
        option_type='put'
    )
    print(f"Calculated Put Option Price: ${put_price:.2f}")

if __name__ == '__main__':
    main()
