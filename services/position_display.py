"""
Position Display Helper

Groups spread legs together for Robinhood-style display.
"""

from typing import List, Dict, Any


def group_spread_positions(positions: List[Dict]) -> List[Dict]:
    """
    Group option spread legs into single combined positions.
    
    Args:
        positions: List of individual positions from Tradier
        
    Returns:
        List of positions with spreads grouped together
    """
    if not positions:
        return []
    
    # Separate option and stock positions
    option_positions = []
    stock_positions = []
    
    for p in positions:
        symbol = p.get('symbol', '')
        # Options have longer symbols (OCC format)
        if len(symbol) > 10:
            option_positions.append(p)
        else:
            stock_positions.append(p)
    
    # Group options by underlying, expiration, and type
    option_groups = {}
    
    for pos in option_positions:
        underlying = pos.get('underlying', '')
        expiration = pos.get('expiration', '')
        option_type = pos.get('option_type', '')
        
        if not underlying or not expiration or not option_type:
            continue
        
        # Create group key
        group_key = f"{underlying}_{expiration}_{option_type}"
        
        if group_key not in option_groups:
            option_groups[group_key] = []
        option_groups[group_key].append(pos)
    
    # Process groups to identify spreads
    combined_positions = []
    processed_symbols = set()
    
    for group_key, group_positions in option_groups.items():
        # Sort by strike price for easier pairing
        group_positions.sort(key=lambda p: p.get('strike', 0))
        
        # Try to pair spreads within this group
        unmatched = []
        
        for pos in group_positions:
            if pos['symbol'] in processed_symbols:
                continue
            
            qty = pos.get('quantity', 0)
            strike = pos.get('strike', 0)
            
            # Look for opposite position (opposite quantity sign, different strike)
            matched = False
            for other_pos in group_positions:
                if other_pos['symbol'] in processed_symbols or other_pos['symbol'] == pos['symbol']:
                    continue
                
                other_qty = other_pos.get('quantity', 0)
                other_strike = other_pos.get('strike', 0)
                
                # Check if opposite quantities and different strikes
                if strike != other_strike and ((qty > 0 and other_qty < 0) or (qty < 0 and other_qty > 0)):
                    # Create spread
                    short_leg = pos if qty < 0 else other_pos
                    long_leg = pos if qty > 0 else other_pos
                    
                    combined = create_spread_position(short_leg, long_leg)
                    combined_positions.append(combined)
                    
                    processed_symbols.add(pos['symbol'])
                    processed_symbols.add(other_pos['symbol'])
                    matched = True
                    break
            
            if not matched and pos['symbol'] not in processed_symbols:
                unmatched.append(pos)
        
        # Add unmatched positions individually
        for pos in unmatched:
            if pos['symbol'] not in processed_symbols:
                combined_positions.append(pos)
                processed_symbols.add(pos['symbol'])
    
    # Add stock positions
    combined_positions.extend(stock_positions)
    
    return combined_positions


def create_spread_position(short_leg: Dict, long_leg: Dict) -> Dict:
    """
    Create a combined spread position from two legs.
    
    Args:
        short_leg: Short option position (negative quantity)
        long_leg: Long option position (positive quantity)
        
    Returns:
        Combined spread position dict
    """
    underlying = short_leg.get('underlying') or long_leg.get('underlying')
    expiration = short_leg.get('expiration') or long_leg.get('expiration')
    option_type = short_leg.get('option_type') or long_leg.get('option_type')
    
    short_strike = short_leg.get('strike', 0)
    long_strike = long_leg.get('strike', 0)
    
    # Determine spread width (always positive)
    width = abs(short_strike - long_strike)
    
    # Spread name
    if option_type == 'put':
        spread_name = f"{underlying} ${short_strike}/{long_strike} Put Spread"
    else:
        spread_name = f"{underlying} ${short_strike}/{long_strike} Call Spread"
    
    # Calculate combined metrics
    quantity = abs(short_leg.get('quantity', 0))
    
    # Entry prices from Tradier are already in per-contract format
    short_entry_total = abs(short_leg.get('entry_price', 0))
    long_entry_total = abs(long_leg.get('entry_price', 0))
    
    # Credit per contract = difference between sell and buy
    # Example: $1280 - $1110 = $170 per contract
    credit_per_contract = short_entry_total - long_entry_total
    
    # Total net credit = credit per contract × number of contracts
    # Example: $170 × 5 contracts = $850 total
    net_credit = credit_per_contract * quantity
    
    # Debug output
    print(f"\n=== SPREAD CALCULATION DEBUG ===")
    print(f"Underlying: {underlying} {short_strike}/{long_strike} {option_type}")
    print(f"Short entry: ${short_entry_total:.2f} per contract (${short_entry_total/100:.2f} per share)")
    print(f"Long entry: ${long_entry_total:.2f} per contract (${long_entry_total/100:.2f} per share)")
    print(f"Credit per contract: ${credit_per_contract:.2f}")
    print(f"Quantity: {quantity}")
    print(f"Net credit: ${net_credit:.2f}")
    
    # Current prices for closing the spread (per-share from API)
    # Short leg: We need to BUY to close, so use ASK price
    # Long leg: We need to SELL to close, so use BID price
    short_ask_per_share = abs(short_leg.get('ask_price', short_leg.get('current_price', 0)))
    long_bid_per_share = abs(long_leg.get('bid_price', long_leg.get('current_price', 0)))
    
    # Convert to per-contract values
    short_ask_total = short_ask_per_share * 100
    long_bid_total = long_bid_per_share * 100
    
    # Current spread value (what it would cost to close)
    # Cost to close = (buy back short at ask) - (sell long at bid) × quantity
    current_spread_value = (short_ask_total - long_bid_total) * quantity
    
    print(f"Short ask: ${short_ask_per_share:.2f} per share (${short_ask_total:.2f} per contract)")
    print(f"Long bid: ${long_bid_per_share:.2f} per share (${long_bid_total:.2f} per contract)")
    print(f"Current spread value: ${current_spread_value:.2f}")
    
    # P&L: Credit received - current value to close
    # Positive = profit, Negative = loss
    pl_dollars = net_credit - current_spread_value
    pl_percent = (pl_dollars / abs(net_credit) * 100) if net_credit != 0 else 0
    
    # For a credit spread:
    # Max Profit = Net credit received
    # Max Loss = (Width × 100) - Credit per contract, then × Quantity
    max_profit = abs(net_credit)
    max_loss = abs((width * 100 - credit_per_contract) * quantity)
    
    print(f"Max Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")
    print(f"================================\n")
    
    return {
        'symbol': spread_name,
        'is_spread': True,
        'underlying': underlying,
        'expiration': expiration,
        'option_type': option_type,
        'quantity': quantity,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'width': width,
        'entry_price': abs(credit_per_contract / 100),  # Per-share equivalent for display
        'current_price': abs(current_spread_value / quantity / 100) if quantity > 0 else 0,
        'pl_dollars': pl_dollars,
        'pl_percent': pl_percent,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'net_credit': abs(net_credit),
        'short_leg': short_leg,
        'long_leg': long_leg,
        'dte': short_leg.get('dte') or long_leg.get('dte')
    }
