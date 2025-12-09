from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from datetime import datetime
from database import Base


class Position(Base):
    """Historical position records"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    underlying = Column(String(20), index=True)
    option_type = Column(String(10))  # 'call' or 'put'
    strike = Column(Float)
    expiration = Column(String(20))
    quantity = Column(Integer)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    entry_date = Column(DateTime, default=datetime.utcnow)
    exit_date = Column(DateTime, nullable=True)
    pl_amount = Column(Float, nullable=True)
    pl_percent = Column(Float, nullable=True)
    status = Column(String(20), default='open')  # 'open', 'closed', 'rolled'
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Order(Base):
    """Order history"""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, index=True)  # Tradier order ID
    symbol = Column(String(50), nullable=False, index=True)
    order_type = Column(String(20))  # 'market', 'limit', 'credit', 'debit'
    order_class = Column(String(20))  # 'option', 'multileg', 'stock'
    side = Column(String(20))  # 'buy_to_open', 'sell_to_close', etc.
    quantity = Column(Integer)
    price = Column(Float, nullable=True)
    status = Column(String(20), index=True)  # 'pending', 'filled', 'cancelled', 'rejected'
    legs = Column(JSON, nullable=True)  # For multileg orders
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SupportResistance(Base):
    """Cached support and resistance levels"""
    __tablename__ = 'support_resistance'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # '1m', '3m', '6m', '1y'
    support_levels = Column(JSON)  # Array of support prices
    resistance_levels = Column(JSON)  # Array of resistance prices
    calculated_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True)
    
    def is_expired(self):
        """Check if cached data is expired"""
        if self.expires_at is None:
            return True
        return datetime.utcnow() > self.expires_at


class MLPrediction(Base):
    """Machine learning prediction history"""
    __tablename__ = 'ml_predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    target_date = Column(DateTime, nullable=False)  # Date of prediction target
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)  # Filled in later
    model_version = Column(String(50))
    features_used = Column(JSON)  # List of feature names
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PositionState(Base):
    """Position state tracking (replaces position_state.json)"""
    __tablename__ = 'position_state'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    itm_consecutive_days = Column(Integer, default=0)
    last_check_date = Column(String(20))  # YYYY-MM-DD format
    additional_metadata = Column(JSON, nullable=True)  # Additional state data
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
