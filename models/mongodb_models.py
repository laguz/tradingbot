"""
MongoDB Collection Models

Provides MongoDB collection interfaces to replace SQLAlchemy models.
Each model class provides helper methods for CRUD operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from database import get_mongo_db
from utils.logger import logger


class MLPredictionModel:
    """Machine learning prediction history - MongoDB collection."""
    
    COLLECTION_NAME = 'ml_predictions'
    
    @staticmethod
    def get_collection():
        """Get the MLPrediction collection."""
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[MLPredictionModel.COLLECTION_NAME]
    
    @staticmethod
    def insert(symbol: str, prediction_date: datetime, target_date: datetime,
               predicted_price: float, model_version: str, features_used: List[str],
               confidence: float = None, actual_price: float = None) -> str:
        """
        Insert a new ML prediction.
        
        Returns:
            Inserted document ID as string
        """
        return MLPredictionModel.upsert_many([{
            'symbol': symbol,
            'prediction_date': prediction_date,
            'target_date': target_date,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'model_version': model_version,
            'features_used': features_used,
            'confidence': confidence,
            'created_at': datetime.utcnow()
        }])
    
    @staticmethod
    def upsert_many(predictions: List[Dict]) -> int:
        """
        Upsert multiple predictions (overwrite if exists for symbol+target_date).
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Number of documents modified/upserted
            
        Note:
            - actual_price will NOT be overwritten if already set in the DB.
            - Use update_actual_prices_bulk() to backfill actual prices.
        """
        collection = MLPredictionModel.get_collection()
        from pymongo import UpdateOne
        
        operations = []
        for pred in predictions:
            # Add created_at if missing
            if 'created_at' not in pred:
                pred['created_at'] = datetime.utcnow()
                
            # Filter by symbol and target_date to ensure uniqueness per day
            filter_query = {
                'symbol': pred['symbol'],
                'target_date': pred['target_date']
            }
            
            # Build $set - exclude actual_price entirely to prevent overwrites
            set_fields = {k: v for k, v in pred.items() 
                         if k not in ['actual_price', 'created_at']}
            
            # Build $setOnInsert - set these only on new documents
            set_on_insert = {'created_at': pred['created_at']}
            
            # Only set actual_price on insert, never on update
            if 'actual_price' in pred:
                set_on_insert['actual_price'] = pred['actual_price']
            else:
                set_on_insert['actual_price'] = None
            
            update_query = {
                '$set': set_fields,
                '$setOnInsert': set_on_insert
            }
            
            operations.append(UpdateOne(filter_query, update_query, upsert=True))
        
        if not operations:
            return 0
            
        result = collection.bulk_write(operations)
        logger.debug(f"Upserted {len(predictions)} predictions: {result.upserted_count} inserts, {result.modified_count} updates")
        return result.upserted_count + result.modified_count
    
    @staticmethod
    def find_by_symbol(symbol: str, limit: int = 100) -> List[Dict]:
        """Find predictions for a symbol, most recent first."""
        collection = MLPredictionModel.get_collection()
        cursor = collection.find(
            {'symbol': symbol}
        ).sort('prediction_date', -1).limit(limit)
        return list(cursor)
    
    @staticmethod
    def find_recent(days: int = 30, limit: int = 1000) -> List[Dict]:
        """Find predictions from the last N days."""
        collection = MLPredictionModel.get_collection()
        cutoff_date = datetime.utcnow()
        cutoff_date = cutoff_date - timedelta(days=days)
        
        cursor = collection.find(
            {'prediction_date': {'$gte': cutoff_date}}
        ).sort('prediction_date', -1).limit(limit)
        return list(cursor)
    
    @staticmethod
    def find_for_correction(symbol: str, target_date: datetime) -> List[Dict]:
        """Find predictions that need actual price updates."""
        collection = MLPredictionModel.get_collection()
        cursor = collection.find({
            'symbol': symbol,
            'target_date': target_date,
            'actual_price': None
        })
        return list(cursor)
    
    @staticmethod
    def update_actual_price(prediction_id: str, actual_price: float) -> bool:
        """Update the actual price for a prediction."""
        from bson import ObjectId
        from bson.errors import InvalidId
        collection = MLPredictionModel.get_collection()
        
        try:
            oid = ObjectId(prediction_id)
        except (InvalidId, TypeError, ValueError):
            raise ValueError(f"Invalid prediction_id: {prediction_id}")
        
        result = collection.update_one(
            {'_id': oid},
            {'$set': {'actual_price': actual_price}}
        )
        return result.modified_count > 0
    
    @staticmethod
    def update_actual_prices_bulk(updates: List[Dict]) -> int:
        """Bulk update actual prices. Updates dict: {id, actual_price}"""
        from bson import ObjectId
        from bson.errors import InvalidId
        collection = MLPredictionModel.get_collection()
        
        from pymongo import UpdateOne
        operations = []
        
        for update in updates:
            try:
                oid = ObjectId(update['id'])
                operations.append(
                    UpdateOne(
                        {'_id': oid},
                        {'$set': {'actual_price': update['actual_price']}}
                    )
                )
            except (InvalidId, TypeError, ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid update: {update} - {e}")
                continue
        
        if not operations:
            return 0
        
        result = collection.bulk_write(operations)
        return result.modified_count
    
    @staticmethod
    def get_stats_by_symbol(symbol: str) -> Dict:
        """Get prediction statistics for a symbol."""
        collection = MLPredictionModel.get_collection()
        
        pipeline = [
            {'$match': {'symbol': symbol, 'actual_price': {'$ne': None}}},
            {'$group': {
                '_id': '$symbol',
                'count': {'$sum': 1},
                'avg_predicted': {'$avg': '$predicted_price'},
                'avg_actual': {'$avg': '$actual_price'}
            }}
        ]
        
        result = list(collection.aggregate(pipeline))
        return result[0] if result else {}


class PositionModel:
    """Historical position records - MongoDB collection."""
    
    COLLECTION_NAME = 'positions'
    
    @staticmethod
    def get_collection():
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[PositionModel.COLLECTION_NAME]
    
    @staticmethod
    def insert(symbol: str, underlying: str, option_type: str, strike: float,
               expiration: str, quantity: int, entry_price: float, **kwargs) -> str:
        """Insert a new position."""
        collection = PositionModel.get_collection()
        
        doc = {
            'symbol': symbol,
            'underlying': underlying,
            'option_type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': kwargs.get('exit_price'),
            'entry_date': kwargs.get('entry_date', datetime.utcnow()),
            'exit_date': kwargs.get('exit_date'),
            'pl_amount': kwargs.get('pl_amount'),
            'pl_percent': kwargs.get('pl_percent'),
            'status': kwargs.get('status', 'open'),
            'notes': kwargs.get('notes'),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = collection.insert_one(doc)
        return str(result.inserted_id)
    
    @staticmethod
    def find_open_positions() -> List[Dict]:
        """Find all open positions."""
        collection = PositionModel.get_collection()
        cursor = collection.find({'status': 'open'}).sort('entry_date', -1)
        return list(cursor)


class OrderModel:
    """Order history - MongoDB collection."""
    
    COLLECTION_NAME = 'orders'
    
    @staticmethod
    def get_collection():
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[OrderModel.COLLECTION_NAME]


class SupportResistanceModel:
    """Cached support and resistance levels - MongoDB collection."""
    
    COLLECTION_NAME = 'support_resistance'
    
    @staticmethod
    def get_collection():
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[SupportResistanceModel.COLLECTION_NAME]
    
    @staticmethod
    def upsert(symbol: str, timeframe: str, support_levels: List[float],
               resistance_levels: List[float], expires_at: datetime = None) -> str:
        """Insert or update S/R levels for a symbol/timeframe."""
        collection = SupportResistanceModel.get_collection()
        
        doc = {
            'symbol': symbol,
            'timeframe': timeframe,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'calculated_at': datetime.utcnow(),
            'expires_at': expires_at
        }
        
        result = collection.update_one(
            {'symbol': symbol, 'timeframe': timeframe},
            {'$set': doc},
            upsert=True
        )
        
        return str(result.upserted_id) if result.upserted_id else "updated"
    
    @staticmethod
    def find_by_symbol_timeframe(symbol: str, timeframe: str) -> Optional[Dict]:
        """Find cached S/R levels."""
        collection = SupportResistanceModel.get_collection()
        return collection.find_one({'symbol': symbol, 'timeframe': timeframe})


class PositionStateModel:
    """Position state tracking - MongoDB collection."""
    
    COLLECTION_NAME = 'position_state'
    
    @staticmethod
    def get_collection():
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[PositionStateModel.COLLECTION_NAME]
    
    @staticmethod
    def upsert(symbol: str, itm_consecutive_days: int = 0,
               last_check_date: str = None, additional_metadata: Dict = None) -> str:
        """Insert or update position state."""
        collection = PositionStateModel.get_collection()
        
        doc = {
            'symbol': symbol,
            'itm_consecutive_days': itm_consecutive_days,
            'last_check_date': last_check_date,
            'additional_metadata': additional_metadata or {},
            'updated_at': datetime.utcnow()
        }
        
        result = collection.update_one(
            {'symbol': symbol},
            {'$set': doc},
            upsert=True
        )
        
        return str(result.upserted_id) if result.upserted_id else "updated"
    
    @staticmethod
    def find_by_symbol(symbol: str) -> Optional[Dict]:
        """Find position state for a symbol."""
        collection = PositionStateModel.get_collection()
        return collection.find_one({'symbol': symbol})


class StockDataModel:
    """Stock OHLCV historical data - MongoDB collection."""
    
    COLLECTION_NAME = 'stock_ohlcv_data'
    
    @staticmethod
    def get_collection():
        """Get the stock OHLCV data collection."""
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[StockDataModel.COLLECTION_NAME]
    
    @staticmethod
    def insert_many(records: List[Dict]) -> List[str]:
        """
        Insert multiple OHLCV records.
        
        Args:
            records: List of dicts with keys: symbol, date, open, high, low, close, volume
            
        Returns:
            List of inserted document IDs (empty list if all duplicates)
        """
        from pymongo.errors import BulkWriteError
        collection = StockDataModel.get_collection()
        
        # Add last_updated timestamp to all records
        for record in records:
            if 'last_updated' not in record:
                record['last_updated'] = datetime.utcnow()
        
        try:
            result = collection.insert_many(records, ordered=False)
            logger.debug(f"Inserted {len(result.inserted_ids)} OHLCV records")
            return [str(id) for id in result.inserted_ids]
        except BulkWriteError as e:
            # Check if errors are only duplicate key violations (code 11000)
            write_errors = e.details.get('writeErrors', [])
            if write_errors and all(err.get('code') == 11000 for err in write_errors):
                inserted_count = e.details.get('nInserted', 0)
                logger.debug(f"Inserted {inserted_count} records, {len(write_errors)} duplicates skipped")
                # Get IDs of successfully inserted documents from write result
                # Note: insertedIds may not be available in error case, use nInserted as indicator
                return []  # Return empty list when duplicates encountered
            # Re-raise if there are other types of errors
            raise
    
    @staticmethod
    def find_by_symbol(symbol: str, start_date: datetime = None, 
                       end_date: datetime = None, limit: int = None) -> List[Dict]:
        """
        Find OHLCV data for a symbol within date range.
        
        Args:
            symbol: Stock ticker
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of records
            
        Returns:
            List of OHLCV records sorted by date ascending
        """
        collection = StockDataModel.get_collection()
        
        query = {'symbol': symbol}
        if start_date or end_date:
            query['date'] = {}
            if start_date:
                query['date']['$gte'] = start_date
            if end_date:
                query['date']['$lte'] = end_date
        
        cursor = collection.find(query).sort('date', 1)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    @staticmethod
    def get_latest_date(symbol: str) -> Optional[datetime]:
        """
        Get the most recent date for which we have data for a symbol.
        """
        collection = StockDataModel.get_collection()
        result = collection.find_one({'symbol': symbol}, sort=[('date', -1)])
        return result['date'] if result else None
    
    @staticmethod
    def get_earliest_date(symbol: str) -> Optional[datetime]:
        """
        Get the oldest date for which we have data for a symbol.
        """
        collection = StockDataModel.get_collection()
        result = collection.find_one({'symbol': symbol}, sort=[('date', 1)])
        return result['date'] if result else None
    
    @staticmethod
    def upsert_daily(symbol: str, date: datetime, open_price: float,
                     high: float, low: float, close: float, volume: float) -> bool:
        """
        Insert or update a single daily bar.
        
        Args:
            symbol: Stock ticker
            date: Trading date
            open_price, high, low, close, volume: OHLCV values
            
        Returns:
            True if inserted/updated successfully
        """
        collection = StockDataModel.get_collection()
        
        doc = {
            'symbol': symbol,
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'last_updated': datetime.utcnow()
        }
        
        result = collection.update_one(
            {'symbol': symbol, 'date': date},
            {'$set': doc},
            upsert=True
        )
        
        return result.acknowledged
    
    @staticmethod
    def count_records(symbol: str) -> int:
        """Count total records for a symbol."""
        collection = StockDataModel.get_collection()
        return collection.count_documents({'symbol': symbol})
    
    @staticmethod
    def get_close_price(symbol: str, date: datetime) -> Optional[float]:
        """
        Get closing price for a specific date.
        
        Args:
            symbol: Stock ticker
            date: Date to check (normalized to midnight for exact match)
            
        Returns:
            Closing price or None
        """
        collection = StockDataModel.get_collection()
        
        # Normalize date to midnight for exact match (faster than range query)
        normalized_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Try exact match first (uses compound index)
        result = collection.find_one({
            'symbol': symbol,
            'date': normalized_date
        })
        
        # Fallback to range query if exact match fails (handles non-normalized dates)
        if not result:
            end_date = normalized_date + timedelta(days=1)
            result = collection.find_one({
                'symbol': symbol,
                'date': {'$gte': normalized_date, '$lt': end_date}
            })
        
        return result['close'] if result else None

    @staticmethod
    def _DANGEROUS_delete_all_data(symbol: str, confirm: str = None) -> int:
        """
        ⚠️ DANGEROUS: Delete ALL OHLCV records for a symbol.
        
        This method permanently deletes historical data. Only use for:
        - Testing/development data cleanup
        - Fixing corrupt data that needs complete refresh
        
        Args:
            symbol: Stock ticker
            confirm: Must be exactly 'DELETE_ALL_DATA' to proceed
            
        Returns:
            Count of deleted records
            
        Raises:
            ValueError: If confirmation string is incorrect
        """
        if confirm != 'DELETE_ALL_DATA':
            raise ValueError(
                "Dangerous operation requires confirm='DELETE_ALL_DATA'. "
                "This will permanently delete all historical data for the symbol."
            )
        
        collection = StockDataModel.get_collection()
        logger.warning(f"⚠️ DELETING ALL OHLCV DATA FOR {symbol}")
        result = collection.delete_many({'symbol': symbol})
        logger.warning(f"Deleted {result.deleted_count} records for {symbol}")
        return result.deleted_count


class UserProfileModel:
    """User profile data - MongoDB collection."""
    
    COLLECTION_NAME = 'user_profiles'
    
    @staticmethod
    def get_collection():
        """Get the user profiles collection."""
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[UserProfileModel.COLLECTION_NAME]
    
    @staticmethod
    def upsert(pubkey: str, username: str = None, metadata: Dict = None) -> str:
        """
        Insert or update user profile.
        
        Args:
            pubkey: User's Nostr public key
            username: Optional custom username
            metadata: Optional additional metadata
            
        Returns:
            Document ID as string
        """
        collection = UserProfileModel.get_collection()
        
        doc = {
            'pubkey': pubkey,
            'updated_at': datetime.utcnow()
        }
        
        if username is not None:
            doc['username'] = username
        
        if metadata is not None:
            doc['metadata'] = metadata
        
        result = collection.update_one(
            {'pubkey': pubkey},
            {'$set': doc},
            upsert=True
        )
        
        logger.debug(f"Updated profile for pubkey {pubkey[:8]}...")
        return str(result.upserted_id) if result.upserted_id else "updated"
    
    @staticmethod
    def find_by_pubkey(pubkey: str) -> Optional[Dict]:
        """
        Find user profile by public key.
        
        Args:
            pubkey: User's Nostr public key
            
        Returns:
            User profile dict or None
        """
        collection = UserProfileModel.get_collection()
        return collection.find_one({'pubkey': pubkey})
    
    @staticmethod
    def find_by_username(username: str) -> Optional[Dict]:
        """
        Find user profile by username.
        
        Args:
            username: Custom username
            
        Returns:
            User profile dict or None
        """
        collection = UserProfileModel.get_collection()
        return collection.find_one({'username': username})
    
    @staticmethod
    def username_exists(username: str, exclude_pubkey: str = None) -> bool:
        """
        Check if username already exists.
        
        Args:
            username: Username to check
            exclude_pubkey: Optional pubkey to exclude from check (for updates)
            
        Returns:
            True if username exists
        """
        collection = UserProfileModel.get_collection()
        
        query = {'username': username}
        if exclude_pubkey:
            query['pubkey'] = {'$ne': exclude_pubkey}
        
        return collection.count_documents(query) > 0


class AutoTradeModel:
    """Automated trade history - MongoDB collection."""
    
    COLLECTION_NAME = 'auto_trades'
    
    @staticmethod
    def get_collection():
        """Get the auto trades collection."""
        db = get_mongo_db()
        if db is None:
            raise Exception("MongoDB not configured")
        return db[AutoTradeModel.COLLECTION_NAME]
    
    @staticmethod
    def insert(trade_data: Dict) -> str:
        """
        Insert automated trade record.
        
        Args:
            trade_data: Trade details
            
        Returns:
            Document ID
        """
        collection = AutoTradeModel.get_collection()
        
        doc = {
            **trade_data,
            'created_at': datetime.utcnow()
        }
        
        result = collection.insert_one(doc)
        logger.debug(f"Inserted auto trade for {trade_data.get('symbol')}")
        return str(result.inserted_id)
    
    @staticmethod
    def find_recent(days: int = 7, limit: int = 100) -> List[Dict]:
        """
        Find recent automated trades.
        
        Args:
            days: Number of days to look back
            limit: Maximum results
            
        Returns:
            List of trade dicts
        """
        collection = AutoTradeModel.get_collection()
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        cursor = collection.find(
            {'created_at': {'$gte': cutoff}}
        ).sort('created_at', -1).limit(limit)
        
        return list(cursor)
    
    @staticmethod
    def get_stats(days: int = 1) -> Dict:
        """
        Get trading statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Stats dict
        """
        collection = AutoTradeModel.get_collection()
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        trades = list(collection.find({'created_at': {'$gte': cutoff}}))
        
        total = len(trades)
        successful = sum(1 for t in trades if t.get('result', {}).get('success'))
        strategies = {}
        symbols = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            symbol = trade.get('symbol', 'unknown')
            
            strategies[strategy] = strategies.get(strategy, 0) + 1
            symbols[symbol] = symbols.get(symbol, 0) + 1
        
        return {
            'total_trades': total,
            'successful': successful,
            'failed': total - successful,
            'by_strategy': strategies,
            'by_symbol': symbols,
            'period_days': days
        }
