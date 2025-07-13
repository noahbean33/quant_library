import numpy as np

def calculate_present_value(fv: float, r: float, t: float, m: int = 1) -> float:
    """
    Calculates the present value of a future cash flow.

    Args:
        fv (float): Future value of the cash flow.
        r (float): Annual interest rate.
        t (float): Time in years.
        m (int, optional): Number of compounding periods per year. 
                           Defaults to 1 (annual compounding). 
                           Use np.inf for continuous compounding.

    Returns:
        float: The present value of the cash flow.
    """
    if m == np.inf:
        # Continuous compounding
        pv = fv * np.exp(-r * t)
    else:
        # Discrete compounding
        pv = fv / (1 + r / m)**(m * t)
    return pv

def calculate_future_value(pv: float, r: float, t: float, m: int = 1) -> float:
    """
    Calculates the future value of a present cash flow.

    Args:
        pv (float): Present value of the cash flow.
        r (float): Annual interest rate.
        t (float): Time in years.
        m (int, optional): Number of compounding periods per year. 
                           Defaults to 1 (annual compounding).
                           Use np.inf for continuous compounding.

    Returns:
        float: The future value of the cash flow.
    """
    if m == np.inf:
        # Continuous compounding
        fv = pv * np.exp(r * t)
    else:
        # Discrete compounding
        fv = pv * (1 + r / m)**(m * t)
    return fv

def calculate_zero_coupon_bond_price(face_value: float, r: float, t: float, m: int = 1) -> float:
    """
    Calculates the price of a zero-coupon bond.

    A zero-coupon bond is a debt security that does not pay interest but is 
    traded at a deep discount, rendering a profit at maturity when the bond 
    is redeemed for its full face value.

    Args:
        face_value (float): The face value of the bond, paid at maturity.
        r (float): The annual market interest rate (discount rate).
        t (float): Time to maturity in years.
        m (int, optional): Number of compounding periods per year. 
                           Defaults to 1 (annual compounding).

    Returns:
        float: The price (present value) of the zero-coupon bond.
    """
    return calculate_present_value(fv=face_value, r=r, t=t, m=m)

def calculate_coupon_bond_price(face_value: float, coupon_rate: float, market_rate: float, years_to_maturity: int, coupon_frequency: int = 1) -> float:
    """
    Calculates the price of a coupon-paying bond.

    The price is the sum of the present values of all future coupon payments 
    plus the present value of the bond's face value.

    Args:
        face_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate of the bond.
        market_rate (float): The current annual market interest rate (yield to maturity).
        years_to_maturity (int): The number of years until the bond matures.
        coupon_frequency (int, optional): The number of coupon payments per year. 
                                         Defaults to 1 (annual).

    Returns:
        float: The market price of the coupon-paying bond.
    """
    # Periodic coupon payment
    coupon_payment = face_value * coupon_rate / coupon_frequency
    
    # Total number of periods
    num_periods = years_to_maturity * coupon_frequency
    
    # Periodic market rate
    periodic_market_rate = market_rate / coupon_frequency
    
    # Calculate the present value of the coupon payments (annuity)
    pv_coupons = 0
    for t in range(1, num_periods + 1):
        pv_coupons += coupon_payment / (1 + periodic_market_rate)**t
        
    # Calculate the present value of the face value
    pv_face_value = face_value / (1 + periodic_market_rate)**num_periods
    
    # The bond price is the sum of the present values
    bond_price = pv_coupons + pv_face_value
    
    return bond_price
