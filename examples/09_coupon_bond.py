import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.valueinvestpy.analysis.fixed_income import calculate_coupon_bond_price

def main():
    """Main function to run the coupon-paying bond pricing example."""
    # 1. Define bond parameters
    face_value = 1000       # The amount paid to the holder at maturity
    coupon_rate = 0.05      # The annual coupon rate (5%)
    market_rate = 0.04      # The current annual market interest rate (4%)
    years_to_maturity = 10  # The number of years until the bond matures

    print("--- Coupon-Paying Bond Pricing ---")
    print(f"Face Value: ${face_value:.2f}")
    print(f"Annual Coupon Rate: {coupon_rate:.2%}")
    print(f"Annual Market Rate (YTM): {market_rate:.2%}")
    print(f"Years to Maturity: {years_to_maturity}")

    # 2. Calculate the price of the bond with annual coupons
    bond_price_annual = calculate_coupon_bond_price(
        face_value=face_value,
        coupon_rate=coupon_rate,
        market_rate=market_rate,
        years_to_maturity=years_to_maturity,
        coupon_frequency=1  # Annual coupons
    )

    # 3. Calculate the price of the bond with semi-annual coupons
    bond_price_semi_annual = calculate_coupon_bond_price(
        face_value=face_value,
        coupon_rate=coupon_rate,
        market_rate=market_rate,
        years_to_maturity=years_to_maturity,
        coupon_frequency=2  # Semi-annual coupons
    )

    # 4. Print the results
    print("\n-------------------------------------")
    print(f"Calculated Price (Annual Coupons): ${bond_price_annual:.2f}")
    print(f"Calculated Price (Semi-Annual Coupons): ${bond_price_semi_annual:.2f}")
    print("-------------------------------------")
    print("Since the coupon rate (5%) is higher than the market rate (4%), the bond trades at a premium.")

if __name__ == '__main__':
    main()
