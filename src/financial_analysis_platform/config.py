# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
NASDAQ_API_KEY = os.getenv("NASDAQ_API_KEY")
INTRINIO_API_KEY = os.getenv("INTRINIO_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
