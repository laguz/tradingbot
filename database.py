"""
MongoDB Database Module

Provides MongoDB connection and database access.
"""

from pymongo import MongoClient
from config import get_config
from utils.logger import logger

config = get_config()

# MongoDB setup
mongo_client = None
mongo_db = None

if config.MONGODB_URI:
    try:
        # Initialize MongoDB client with SSL configuration for development
        mongo_client = MongoClient(
            config.MONGODB_URI,
            tlsAllowInvalidCertificates=True  # For development - handle SSL cert issues
        )
        # Extract database name from URI or use default
        mongo_db = mongo_client.get_database('tradingbot')
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB client: {e}")
        mongo_client = None
        mongo_db = None


def init_db():
    """
    Initialize the database connection.
    Should be called on application startup.
    """
    # Test MongoDB connection if configured
    if mongo_db is not None:
        try:
            # Ping MongoDB to verify connection
            mongo_client.admin.command('ping')
            logger.info(f"MongoDB connected successfully to database: {mongo_db.name}")
            
            # Create indexes for better performance
            _create_indexes()
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
    else:
        logger.warning("MongoDB URI not configured - database features will not work!")


def _create_indexes():
    """Create database indexes for optimal performance."""
    try:
        # MLPrediction indexes
        mongo_db.ml_predictions.create_index([('symbol', 1), ('prediction_date', -1)])
        mongo_db.ml_predictions.create_index([('target_date', 1)])
        mongo_db.ml_predictions.create_index([('symbol', 1), ('target_date', 1)])
        
        # Position indexes
        mongo_db.positions.create_index([('symbol', 1)])
        mongo_db.positions.create_index([('underlying', 1)])
        mongo_db.positions.create_index([('status', 1)])
        
        # Order indexes
        mongo_db.orders.create_index([('order_id', 1)], unique=True, sparse=True)
        mongo_db.orders.create_index([('symbol', 1)])
        mongo_db.orders.create_index([('status', 1)])
        
        # SupportResistance indexes
        mongo_db.support_resistance.create_index([('symbol', 1), ('timeframe', 1)], unique=True)
        mongo_db.support_resistance.create_index([('calculated_at', 1)])
        
        # PositionState indexes
        mongo_db.position_state.create_index([('symbol', 1)], unique=True)
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.warning(f"Could not create some MongoDB indexes: {e}")


def get_mongo_db():
    """
    Get the MongoDB database instance.
    Returns None if MongoDB is not configured.
    
    Usage:
        db = get_mongo_db()
        if db:
            collection = db['my_collection']
            # Use collection
    """
    return mongo_db


def close_connection():
    """Close MongoDB connection."""
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")
