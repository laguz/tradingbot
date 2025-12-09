import time
import functools
from config import get_config
from utils.logger import logger

config = get_config()


def retry_api_call(func):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Retries on exceptions with increasing delays between attempts.
    Uses configuration from config.py for retry parameters.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        attempts = config.API_RETRY_ATTEMPTS
        delay = config.API_RETRY_DELAY
        
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == attempts:
                    logger.error(f"{func.__name__} failed after {attempts} attempts: {e}")
                    raise
                
                wait_time = min(delay * (config.API_RETRY_BACKOFF ** (attempt - 1)), 
                              config.API_RETRY_MAX_DELAY)
                
                logger.warning(
                    f"{func.__name__} failed (attempt {attempt}/{attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                time.sleep(wait_time)
        
        # Should never reach here
        return None
    
    return wrapper


class RetryableRequest:
    """
    Context manager for retrying requests with better control.
    
    Usage:
        with RetryableRequest(max_attempts=3) as retry:
            response = retry.execute(requests.get, url, headers=headers)
    """
    
    def __init__(self, max_attempts=None, delay=None, backoff=None, max_delay=None):
        self.max_attempts = max_attempts or config.API_RETRY_ATTEMPTS
        self.delay = delay or config.API_RETRY_DELAY
        self.backoff = backoff or config.API_RETRY_BACKOFF
        self.max_delay = max_delay or config.API_RETRY_MAX_DELAY
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def execute(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_attempts:
                    logger.error(f"Request failed after {self.max_attempts} attempts: {e}")
                    raise
                
                wait_time = min(self.delay * (self.backoff ** (attempt - 1)), 
                              self.max_delay)
                
                logger.warning(
                    f"Request failed (attempt {attempt}/{self.max_attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                time.sleep(wait_time)
