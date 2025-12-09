from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config import get_config

config = get_config()

# Create SQLAlchemy engine
engine = create_engine(
    config.DATABASE_URL,
    echo=config.DEBUG,  # Log SQL queries in debug mode
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=3600    # Recycle connections after 1 hour
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Base class for all models
Base = declarative_base()


def init_db():
    """
    Initialize the database by creating all tables.
    Should be called on application startup.
    """
    from models.db_models import Position, Order, SupportResistance, MLPrediction, PositionState
    Base.metadata.create_all(engine)


def get_session():
    """
    Get a database session.
    Usage:
        session = get_session()
        try:
            # Use session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    """
    return Session()


def close_session():
    """Remove the current session"""
    Session.remove()
