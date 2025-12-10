"""
List all open positions to debug position count discrepancy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tradier_service import get_open_positions
from utils.logger import logger

def list_all_positions():
    """List all open positions from Tradier API."""
    
    print("=" * 80)
    print("FETCHING OPEN POSITIONS FROM TRADIER API")
    print("=" * 80)
    
    positions = get_open_positions()
    
    if positions is None:
        print("❌ Error fetching positions from Tradier API")
        return
    
    if isinstance(positions, dict) and 'error' in positions:
        print(f"❌ Error: {positions['error']}")
        return
    
    if not positions:
        print("✅ No open positions found")
        return
    
    print(f"\n📊 Total Positions: {len(positions)}")
    print("=" * 80)
    
    # Group positions by underlying
    from collections import defaultdict
    by_underlying = defaultdict(list)
    
    for pos in positions:
        underlying = pos.get('underlying') or pos['symbol']
        by_underlying[underlying].append(pos)
    
    # Display positions grouped by underlying
    for underlying, pos_list in sorted(by_underlying.items()):
        print(f"\n{underlying}: {len(pos_list)} position(s)")
        print("-" * 80)
        
        for i, pos in enumerate(pos_list, 1):
            symbol = pos['symbol']
            qty = pos['quantity']
            dte = pos.get('dte', 'N/A')
            pl = pos.get('pl_dollars', 0)
            pl_pct = pos.get('pl_percent', 0)
            
            print(f"  {i}. {symbol}")
            print(f"     Quantity: {qty}")
            print(f"     DTE: {dte}")
            print(f"     P/L: ${pl:.2f} ({pl_pct:+.2f}%)")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(positions)} positions")
    print("=" * 80)
    
    # Ask user if they want to see full details
    print("\n💡 If you see positions here that shouldn't exist, they are coming")
    print("   directly from Tradier's API. You need to close them via Tradier")
    print("   or check if they are expired positions not yet settled.")

if __name__ == '__main__':
    list_all_positions()
