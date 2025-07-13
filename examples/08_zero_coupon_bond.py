import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.fixed_income import calculate_zero_coupon_bond_price

def main():
    """Main function to run the zero-coupon bond pricing example."""
    # 1. Define bond parameters
    face_value = 1000  # The amount paid to the holder at maturity
    interest_rate = 0.04  # The annual market interest rate (discount rate)
    years_to_maturity = 5  # The number of years until the bond matures

    print("--- Zero-Coupon Bond Pricing ---")
    print(f"Face Value: ${face_value:.2f}")
    print(f"Annual Market Rate: {interest_rate:.2%}")
    print(f"Years to Maturity: {years_to_maturity}")

    # 2. Calculate the price of the bond
    # We assume annual compounding, which is standard for this type of calculation.
    bond_price = calculate_zero_coupon_bond_price(
        face_value=face_value,
        r=interest_rate,
        t=years_to_maturity,
        m=1  # Annual compounding
    )

    # 3. Print the result
    print("\n-------------------------------------")
    print(f"Calculated Bond Price: ${bond_price:.2f}")
    print("-------------------------------------")
    print("This is the price an investor should pay today to earn the market rate of 4% over 5 years.")

if __name__ == '__main__':
    main()
