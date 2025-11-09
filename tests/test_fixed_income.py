import unittest
import numpy as np
from src.analysis.fixed_income import (
    calculate_present_value, 
    calculate_future_value, 
    calculate_zero_coupon_bond_price,
    calculate_coupon_bond_price
)

class TestFixedIncome(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the tests."""
        self.value = 1000
        self.r = 0.05
        self.t = 5

    def test_calculate_present_value(self):
        """Test present value calculations for various compounding frequencies."""
        # Annual compounding
        pv_annual = calculate_present_value(self.value, self.r, self.t, m=1)
        self.assertAlmostEqual(pv_annual, 783.53, places=2)

        # Semi-annual compounding
        pv_semi_annual = calculate_present_value(self.value, self.r, self.t, m=2)
        self.assertAlmostEqual(pv_semi_annual, 781.20, places=2)

        # Continuous compounding
        pv_continuous = calculate_present_value(self.value, self.r, self.t, m=np.inf)
        self.assertAlmostEqual(pv_continuous, 778.80, places=2)

    def test_calculate_future_value(self):
        """Test future value calculations for various compounding frequencies."""
        # Annual compounding
        fv_annual = calculate_future_value(self.value, self.r, self.t, m=1)
        self.assertAlmostEqual(fv_annual, 1276.28, places=2)

        # Semi-annual compounding
        fv_semi_annual = calculate_future_value(self.value, self.r, self.t, m=2)
        self.assertAlmostEqual(fv_semi_annual, 1280.08, places=2)

        # Continuous compounding
        fv_continuous = calculate_future_value(self.value, self.r, self.t, m=np.inf)
        self.assertAlmostEqual(fv_continuous, 1284.03, places=2)

    def test_calculate_zero_coupon_bond_price(self):
        """Test the zero-coupon bond price calculation."""
        # Using parameters from the example script
        face_value = 1000
        r = 0.04
        t = 5

        # Test with annual compounding
        price_annual = calculate_zero_coupon_bond_price(face_value, r, t, m=1)
        self.assertAlmostEqual(price_annual, 821.93, places=2)

        # Test with semi-annual compounding
        price_semi_annual = calculate_zero_coupon_bond_price(face_value, r, t, m=2)
        self.assertAlmostEqual(price_semi_annual, 820.35, places=2)


    def test_calculate_coupon_bond_price(self):
        """Test the coupon-paying bond price calculation."""
        # Using parameters from the example script
        face_value = 1000
        coupon_rate = 0.05
        market_rate = 0.04
        years_to_maturity = 10

        # Test with annual coupons
        price_annual = calculate_coupon_bond_price(face_value, coupon_rate, market_rate, years_to_maturity, 1)
        self.assertAlmostEqual(price_annual, 1081.11, places=2)

        # Test with semi-annual coupons
        price_semi_annual = calculate_coupon_bond_price(face_value, coupon_rate, market_rate, years_to_maturity, 2)
        self.assertAlmostEqual(price_semi_annual, 1081.76, places=2)


if __name__ == '__main__':
    unittest.main()
