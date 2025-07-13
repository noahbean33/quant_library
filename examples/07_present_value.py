import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.fixed_income import calculate_present_value, calculate_future_value

def main():
    """Main function to run the present and future value calculation examples."""
    # 1. Define parameters
    cash_flow = 1000  # A cash flow of $1,000
    rate = 0.05       # 5% annual interest rate
    years = 5         # 5-year period

    print("--- Present & Future Value Calculations ---")
    print(f"Cash Flow: ${cash_flow:.2f}")
    print(f"Annual Interest Rate: {rate:.2%}")
    print(f"Period: {years} years")

    # 2. Calculate Present Value with different compounding
    print("\n--- Present Value of $1,000 in 5 years ---")
    
    pv_annual = calculate_present_value(fv=cash_flow, r=rate, t=years, m=1)
    print(f"Annually Compounded: ${pv_annual:.2f}")

    pv_semi_annual = calculate_present_value(fv=cash_flow, r=rate, t=years, m=2)
    print(f"Semi-Annually Compounded: ${pv_semi_annual:.2f}")

    pv_continuous = calculate_present_value(fv=cash_flow, r=rate, t=years, m=np.inf)
    print(f"Continuously Compounded: ${pv_continuous:.2f}")

    # 3. Calculate Future Value with different compounding
    print("\n--- Future Value of $1,000 today ---")

    fv_annual = calculate_future_value(pv=cash_flow, r=rate, t=years, m=1)
    print(f"Annually Compounded: ${fv_annual:.2f}")

    fv_semi_annual = calculate_future_value(pv=cash_flow, r=rate, t=years, m=2)
    print(f"Semi-Annually Compounded: ${fv_semi_annual:.2f}")

    fv_continuous = calculate_future_value(pv=cash_flow, r=rate, t=years, m=np.inf)
    print(f"Continuously Compounded: ${fv_continuous:.2f}")

if __name__ == '__main__':
    main()
