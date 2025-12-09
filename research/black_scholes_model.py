"""
Black-Scholes Model for Option Probability Calculations

Provides theoretical probability calculations for option strike predictions
using the Black-Scholes framework.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import pandas as pd
from utils.logger import logger


def calculate_historical_volatility(prices, window=252):
    """
    Calculate annualized historical volatility from price series.
    
    Args:
        prices: Series or array of prices
        window: Lookback window (default 252 trading days = 1 year)
        
    Returns:
        Annualized volatility (sigma)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Use last 'window' days or all available
    recent_returns = log_returns[-window:] if len(log_returns) > window else log_returns
    
    # Annualized volatility
    volatility = np.std(recent_returns) * np.sqrt(252)
    
    return volatility


def black_scholes_probability(current_price, strike_price, days_to_expiration, volatility, risk_free_rate=0.05):
    """
    Calculate probability of stock price touching strike before expiration
    using Black-Scholes framework (barrier option approximation).
    
    Args:
        current_price: Current stock price
        strike_price: Target strike price
        days_to_expiration: Days until expiration
        volatility: Annualized volatility (sigma)
        risk_free_rate: Risk-free interest rate (default 5%)
        
    Returns:
        Probability of touching strike (0-1)
    """
    if days_to_expiration <= 0:
        return 1.0 if abs(current_price - strike_price) < 0.01 else 0.0
    
    S = current_price
    K = strike_price
    T = days_to_expiration / 252  # Convert to years
    sigma = volatility
    r = risk_free_rate
    
    # For barrier option (probability of touching)
    # Use reflection principle: P(touch) = 2 * N(d)
    # where d = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T))
    
    if S <= 0 or K <= 0 or sigma <= 0:
        logger.warning(f"Invalid parameters: S={S}, K={K}, sigma={sigma}")
        return 0.5  # Return neutral probability
    
    # Calculate d parameter
    d = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Probability of touching (simplified barrier option)
    # For up-and-out: if strike > current, prob of touching = 2*N(d_barrier)
    # For down-and-out: if strike < current, prob of touching = 2*N(-d_barrier)
    
    if K > S:
        # Upward barrier
        prob_touch = 2 * stats.norm.cdf(-abs(d))
    else:
        # Downward barrier  
        prob_touch = 2 * stats.norm.cdf(-abs(d))
    
    # More accurate formula using barrier option pricing
    # P(touch barrier H) = (S/H)^(2*mu/sigma^2)
    # where mu = r - 0.5*sigma^2
    
    mu = r - 0.5 * sigma**2
    exponent = 2 * mu / (sigma**2)
    
    prob_barrier = (S / K) ** exponent
    
    # Average the two methods for robustness
    prob = (prob_touch + prob_barrier) / 2
    
    # Clip to valid probability range
    prob = np.clip(prob, 0.0, 1.0)
    
    return float(prob)


def calculate_option_greeks(current_price, strike_price, days_to_expiration, volatility, 
                           option_type='call', risk_free_rate=0.05):
    """
    Calculate Black-Scholes option Greeks.
    
    Args:
        current_price: Current stock price
        strike_price: Strike price
        days_to_expiration: Days until expiration
        volatility: Annualized volatility
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free rate
        
    Returns:
        Dict with delta, gamma, theta, vega
    """
    S = current_price
    K = strike_price
    T = days_to_expiration / 252
    sigma = volatility
    r = risk_free_rate
    
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1
    
    # Gamma (same for call and put)
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    term1 = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        theta = (term1 + term2) / 252  # Per day
    else:
        term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        theta = (term1 + term2) / 252  # Per day
    
    # Vega (same for call and put)
    vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega': round(vega, 4)
    }


def get_strike_probabilities_bs(ticker, current_price, strikes, days_to_expiration, historical_prices):
    """
    Get Black-Scholes probabilities for multiple strikes.
    
    Args:
        ticker: Stock ticker
        current_price: Current price
        strikes: List of strike prices
        days_to_expiration: Days to expiration
        historical_prices: Historical price series for volatility calculation
        
    Returns:
        Dict mapping strike to probability
    """
    # Calculate historical volatility
    volatility = calculate_historical_volatility(historical_prices)
    
    logger.info(f"Calculated volatility for {ticker}: {volatility:.2%}")
    
    probabilities = {}
    for strike in strikes:
        prob = black_scholes_probability(
            current_price, 
            strike, 
            days_to_expiration, 
            volatility
        )
        probabilities[strike] = prob
    
    return {
        'probabilities': probabilities,
        'volatility': volatility,
        'current_price': current_price
    }
