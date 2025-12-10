"""
Test Daily P/L tracking functionality.

Verifies:
1. Daily P/L calculation from positions
2. Yearly cumulative P/L calculation
3. Date range queries
4. Yearly summary statistics
"""

from datetime import datetime, timedelta
from models.mongodb_models import DailyPLModel, PositionModel
from utils.logger import logger


def test_daily_pl_tracking():
    """Test complete daily P/L functionality."""
    
    # Clean up test data
    DailyPLModel.get_collection().delete_many({'date': {'$gte': datetime(2025, 1, 1)}})
    PositionModel.get_collection().delete_many({'symbol': 'TEST_PL'})
    
    print("Testing Daily P/L Tracking...\n")
    
    # Test 1: Manual upsert
    print("Test 1: Manual upsert")
    test_date = datetime(2025, 1, 15)
    result = DailyPLModel.upsert_daily(
        date=test_date,
        daily_pl=500.75,
        yearly_cumulative_pl=500.75,
        trade_count=3,
        positions_closed=3,
        notes="Test day"
    )
    
    record = DailyPLModel.get_collection().find_one({'date': test_date.replace(hour=0, minute=0, second=0, microsecond=0)})
    assert record is not None
    assert record['daily_pl'] == 500.75
    assert record['yearly_cumulative_pl'] == 500.75
    print(f"✓ Manual upsert successful: {record['daily_pl']:.2f}\n")
    
    # Test 2: Calculate from positions
    print("Test 2: Calculate from positions")
    
    # Create test positions closed on Jan 16, 2025
    test_date2 = datetime(2025, 1, 16, 14, 30)
    
    positions = [
        {
            'symbol': 'TEST_PL',
            'underlying': 'TSLA',
            'option_type': 'call',
            'strike': 450.0,
            'expiration': '2025-01-17',
            'quantity': 10,
            'entry_price': 5.50,
            'exit_price': 7.00,
            'entry_date': datetime(2025, 1, 10),
            'exit_date': test_date2,
            'pl_amount': 1500.0,  # $1.50 * 100 * 10 contracts
            'pl_percent': 27.27,
            'status': 'closed'
        },
        {
            'symbol': 'TEST_PL',
            'underlying': 'SPY',
            'option_type': 'put',
            'strike': 580.0,
            'expiration': '2025-01-17',
            'quantity': 5,
            'entry_price': 3.00,
            'exit_price': 2.00,
            'entry_date': datetime(2025, 1, 10),
            'exit_date': test_date2,
            'pl_amount': -500.0,  # -$1.00 * 100 * 5 contracts
            'pl_percent': -33.33,
            'status': 'closed'
        }
    ]
    
    for pos in positions:
        PositionModel.insert(**pos)
    
    # Calculate P/L for Jan 16
    result = DailyPLModel.calculate_and_store(test_date2)
    
    assert result['daily_pl'] == 1000.0, f"Expected 1000.0, got {result['daily_pl']}"
    assert result['yearly_cumulative_pl'] == 1500.75, f"Expected 1500.75 (500.75 + 1000), got {result['yearly_cumulative_pl']}"
    assert result['positions_closed'] == 2
    print(f"✓ Calculated P/L: Daily=${result['daily_pl']:.2f}, YTD=${result['yearly_cumulative_pl']:.2f}\n")
    
    # Test 3: Date range query
    print("Test 3: Date range query")
    records = DailyPLModel.find_by_date_range(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31)
    )
    
    assert len(records) == 2
    print(f"✓ Found {len(records)} records for January 2025\n")
    
    # Test 4: Get latest yearly P/L
    print("Test 4: Get latest yearly P/L")
    latest_ytd = DailyPLModel.get_latest_yearly_pl()
    assert latest_ytd == 1500.75
    print(f"✓ Latest YTD P/L: ${latest_ytd:.2f}\n")
    
    # Test 5: Yearly summary
    print("Test 5: Yearly summary")
    summary = DailyPLModel.get_yearly_summary(2025)
    
    assert summary['year'] == 2025
    assert summary['total_pl'] == 1500.75
    assert summary['trading_days'] == 2
    assert summary['best_day']['pl'] == 1000.0
    assert summary['average_daily_pl'] == 750.375
    
    print(f"✓ Yearly Summary:")
    print(f"  Year: {summary['year']}")
    print(f"  Total P/L: ${summary['total_pl']:.2f}")
    print(f"  Trading Days: {summary['trading_days']}")
    print(f"  Best Day: ${summary['best_day']['pl']:.2f}")
    print(f"  Worst Day: ${summary['worst_day']['pl']:.2f}")
    print(f"  Average Daily: ${summary['average_daily_pl']:.2f}\n")
    
    # Test 6: Year rollover
    print("Test 6: Year rollover (Jan 1 resets cumulative)")
    jan1_2026 = datetime(2026, 1, 1)
    result = DailyPLModel.calculate_and_store(jan1_2026)
    
    assert result['yearly_cumulative_pl'] == 0.0, "Jan 1 should reset yearly cumulative"
    print(f"✓ Year rollover works: YTD reset to ${result['yearly_cumulative_pl']:.2f}\n")
    
    # Clean up
    DailyPLModel.get_collection().delete_many({'date': {'$gte': datetime(2025, 1, 1)}})
    PositionModel.get_collection().delete_many({'symbol': 'TEST_PL'})
    
    print("✅ All Daily P/L tests passed!")


if __name__ == "__main__":
    test_daily_pl_tracking()
