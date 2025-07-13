# examples/02_broker_connection.py

"""
This example demonstrates how to connect to the Interactive Brokers (IB)
TWS or Gateway application to fetch live account data.

**Prerequisites:**
1.  You must have the Interactive Brokers Trader Workstation (TWS) or
    Gateway application running and logged in.
2.  In TWS/Gateway, go to File -> Global Configuration -> API -> Settings
    and ensure "Enable ActiveX and Socket Clients" is checked.
3.  Note the "Socket port" number from that same settings page. The default
    for TWS is 7497 and for Gateway is 4002.
"""

import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valueinvestpy.trading.broker import InteractiveBrokersBroker

def run_example():
    """
    Connects to IB, fetches positions, and prints them.
    """
    # Connection parameters - adjust if your setup is different
    host = '127.0.0.1'  # Use '127.0.0.1' for a local connection
    port = 7497         # Default for TWS. Use 4002 for Gateway.
    client_id = 1       # A unique ID for this client connection.

    print(f"Attempting to connect to IB TWS/Gateway at {host}:{port} with client ID {client_id}...")

    # 1. Initialize the broker
    ib_broker = InteractiveBrokersBroker(host=host, port=port, client_id=client_id)

    try:
        # 2. Connect to the broker
        ib_broker.connect()

        # 3. Check if connected
        if ib_broker.is_connected():
            print("\nSuccessfully connected to Interactive Brokers!")

            # 4. Fetch your current positions
            print("\nFetching account positions...")
            positions = ib_broker.get_positions()

            if positions:
                print("Current Positions:")
                for pos in positions:
                    print(
                        f"  - Symbol: {pos['symbol']}, "
                        f"Quantity: {pos['quantity']}, "
                        f"Average Cost: {pos['average_cost']}"
                    )
            else:
                print("No positions found in the account.")

        else:
            print("\nFailed to connect. Please ensure TWS/Gateway is running and API settings are correct.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 5. Disconnect when done to release the connection
        if ib_broker and ib_broker.is_connected():
            print("\nDisconnecting from broker...")
            ib_broker.disconnect()
            print("Disconnected.")

if __name__ == '__main__':
    run_example()
