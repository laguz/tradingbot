# Top 10 Features for ML Price Prediction
# Based on feature importance analysis across 6 tickers

TOP_10_FEATURES = [
    'Lag_1',              # 40.5% importance - Previous day's close
    'Distance_SMA_200',   # 13.8% - Distance from 200-day SMA
    'BB_Lower',           # 11.2% - Bollinger Band lower bound
    'OBV',                #  9.2% - On-Balance Volume
    'Lag_2',              #  9.2% - 2 days ago close
    'Lag_3',              #  5.0% - 3 days ago close
    'Lag_4',              #  4.0% - 4 days ago close
    'Lag_5',              #  1.2% - 5 days ago close
    'Signal_Line',        #  1.1% - MACD signal line
    'BB_Upper',           #  1.1% - Bollinger Band upper bound
]
