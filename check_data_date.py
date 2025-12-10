
from models.mongodb_models import StockDataModel
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def check_date():
    ticker = "RIOT"
    latest_date = StockDataModel.get_latest_date(ticker)
    print(f"Latest data for {ticker}: {latest_date}")
    
    if latest_date:
        # Check if we have price for this date
        price = StockDataModel.get_close_price(ticker, latest_date)
        print(f"Close price on {latest_date}: {price}")

if __name__ == "__main__":
    check_date()
