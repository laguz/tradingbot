import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration class"""
    
    # Flask Configuration
    # Use a stable key for dev to allow session persistence across restarts
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-stable-secret-key-change-in-prod')
    DEBUG = False
    
    # Session Configuration
    from datetime import timedelta
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Tradier API Configuration
    TRADIER_API_KEY = os.getenv('TRADIER_API_KEY')
    TRADIER_ACCOUNT_ID = os.getenv('TRADIER_ACCOUNT_ID')
    TRADIER_BASE_URL = 'https://sandbox.tradier.com/v1/'
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///tradingbot.db')
    MONGODB_URI = os.getenv('MONGODB_URI')
    
    # Trading Parameters
    # Auto-Close Rules
    AUTO_CLOSE_DTE_HIGH = 14  # Days to expiration threshold for high DTE
    AUTO_CLOSE_DTE_LOW = 14   # Days to expiration threshold for low DTE
    AUTO_CLOSE_PROFIT_HIGH_DTE = 60.0  # Profit % for DTE >= 14
    AUTO_CLOSE_PROFIT_LOW_DTE = 80.0   # Profit % for DTE < 14
    AUTO_CLOSE_ITM_DTE = 9    # Close ITM positions when DTE < this
    AUTO_CLOSE_ITM_DAYS = 2   # Close ITM after this many consecutive days
    AUTO_CLOSE_ITM_PREMIUM = 0.02  # Premium added to ask for ITM closes
    
    # Auto-Roll Rules
    AUTO_ROLL_DTE = 9         # Roll when DTE < this
    AUTO_ROLL_MIN_PRICE = 1.01  # Roll when option price < this
    AUTO_ROLL_ITM_DAYS = 2    # Roll ITM positions after this many consecutive days
    AUTO_ROLL_MIN_EXP_DAYS = 42  # New expiration must be > this many days
    AUTO_ROLL_STRIKE_ADJUSTMENT = 1  # Adjust strike by this amount
    
    # Support/Resistance Algorithm
    SR_WINDOW = 5             # Window size for pivot detection
    SR_TOLERANCE = 0.015      # Clustering tolerance (1.5%)
    SR_MAX_LEVELS = 5         # Maximum levels to return
    SR_TIMEFRAME = '6m'       # Default timeframe for analysis
    
    # Enhanced S/R with RSI/MACD
    SR_USE_INDICATORS = True  # Use RSI/MACD confirmation
    SR_RSI_OVERSOLD = 30      # RSI oversold threshold
    SR_RSI_OVERBOUGHT = 70    # RSI overbought threshold
    SR_INDICATOR_WEIGHT = 0.3 # Weight for indicator score (0-1)
    
    # Smart Strike Selection
    STRIKE_SAFETY_BUFFER = 0.01  # 1% OTM safety buffer
    STRIKE_FALLBACK_OTM = 0.05   # 5% OTM fallback
    
    # API Retry Configuration
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY = 1.0     # Initial delay in seconds
    API_RETRY_BACKOFF = 2.0   # Exponential backoff multiplier
    API_RETRY_MAX_DELAY = 10.0  # Maximum delay between retries
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_DIR = 'logs'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5
    
    # ML Model Configuration
    ML_MODEL_DIR = 'models'
    ML_RETRAIN_DAYS = 7       # Retrain model every N days
    ML_CACHE_HOURS = 24       # Cache predictions for N hours
    
    # ML Model Hyperparameters
    ML_N_ESTIMATORS = 200     # Number of trees in forest
    ML_MAX_DEPTH = 15         # Maximum tree depth
    ML_MIN_SAMPLES_SPLIT = 5  # Minimum samples to split node
    ML_MIN_SAMPLES_LEAF = 2   # Minimum samples in leaf
    ML_ENABLE_ENSEMBLE = True # Use ensemble vs single RF
    ML_USE_XGBOOST = True     # Include XGBoost in ensemble
    
    # ML Feature Configuration
    ML_ENABLE_NORMALIZATION = True
    ML_ENABLE_VOLATILITY_FEATURES = True
    ML_ENABLE_TIME_FEATURES = True
    ML_ENABLE_MARKET_CONTEXT = False  # VIX, SPY/QQQ correlation (slower)
    ML_LOOKBACK_DAYS = 252 * 2  # 2 years of historical data
    
    # ML Validation
    ML_VALIDATION_SPLITS = 5
    ML_MIN_TRAIN_SIZE = 252   # Minimum 1 year training data
    
    # Automated Trading Configuration
    AUTO_TRADE_ENABLED = False  # Master switch - KEEP FALSE UNTIL TESTED
    AUTO_TRADE_DRY_RUN = True   # Simulation mode (no real orders)
    AUTO_TRADE_MAX_DAILY_LOSS = 500.0  # Stop trading if daily loss exceeds this
    AUTO_TRADE_MAX_POSITION_SIZE = 5   # Maximum contracts per position
    AUTO_TRADE_MAX_POSITIONS = 100     # Maximum open positions at once
    AUTO_TRADE_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']  # Allowed symbols
    AUTO_TRADE_SCHEDULE_HOURS = [10, 14]  # Hours to run (24-hour format, EST)
    AUTO_TRADE_MIN_CONFIDENCE = 0.65   # ML prediction confidence threshold
    AUTO_TRADE_MIN_CREDIT = 0.75       # Minimum credit for spreads
    AUTO_TRADE_SPREAD_WIDTH = 5.0      # Default spread width
    AUTO_TRADE_WHEEL_DTE_MIN = 40      # Minimum DTE for wheel positions
    AUTO_TRADE_WHEEL_DTE_MAX = 50      # Maximum DTE for wheel positions
    AUTO_TRADE_SPREAD_DTE_MIN = 18     # Minimum DTE for credit spreads
    AUTO_TRADE_SPREAD_DTE_MAX = 25     # Maximum DTE for credit spreads
    AUTO_TRADE_NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')  # Optional email alerts



class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///test.db'
    LOG_LEVEL = 'DEBUG'


# Select configuration based on environment
def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
