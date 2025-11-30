import os
from dotenv import load_dotenv
from services.tradier_service import calculate_smart_strikes, get_historical_data

load_dotenv()

def test_smart_strikes():
    symbol = 'TSLA'
    expiration = '2024-06-21' # Adjust as needed
    width = 5
    
    print(f"Testing Smart Strikes for {symbol}...")
    
    # Test Put Credit Spread (Bullish -> Support)
    print("\n--- Put Credit Spread (Support) ---")
    try:
        short, long = calculate_smart_strikes(symbol, expiration, 'credit', 'put', width)
        print(f"Calculated: Short: {short}, Long: {long}")
    except Exception as e:
        print(f"Error: {e}")

    # Test Call Credit Spread (Bearish -> Resistance)
    print("\n--- Call Credit Spread (Resistance) ---")
    try:
        short, long = calculate_smart_strikes(symbol, expiration, 'credit', 'call', width)
        print(f"Calculated: Short: {short}, Long: {long}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Mocking for now since we haven't written the function yet
    pass
