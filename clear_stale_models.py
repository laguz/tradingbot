#!/usr/bin/env python3
"""
Clear stale ML models to force retraining with current data.

This script removes all cached model files (.pkl and .pt) from the ML model directory,
forcing the system to retrain with current market data.
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from utils.logger import logger

config = get_config()


def clear_stale_models(confirm=True):
    """
    Clear all cached ML models.
    
    Args:
        confirm: If True, ask for user confirmation before deletion
        
    Returns:
        Number of files deleted
    """
    model_dir = Path(config.ML_MODEL_DIR)
    
    if not model_dir.exists():
        logger.info(f"Model directory does not exist: {model_dir}")
        return 0
    
    # Find all model files
    pkl_files = list(model_dir.glob("*.pkl"))
    pt_files = list(model_dir.glob("*.pt"))
    all_files = pkl_files + pt_files
    
    if not all_files:
        logger.info("No model files found to delete")
        return 0
    
    print("=" * 80)
    print("CLEAR STALE ML MODELS")
    print("=" * 80)
    print(f"\nModel directory: {model_dir}")
    print(f"\nFound {len(all_files)} model files:")
    print(f"  - {len(pkl_files)} pickle files (.pkl)")
    print(f"  - {len(pt_files)} PyTorch files (.pt)")
    
    # List files
    print("\nFiles to be deleted:")
    for f in all_files[:10]:  # Show first 10
        print(f"  - {f.name}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more")
    
    # Confirmation
    if confirm:
        print("\n" + "⚠️  WARNING: This will force retraining of all models!")
        response = input("\nProceed with deletion? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    # Delete files
    deleted_count = 0
    for file_path in all_files:
        try:
            file_path.unlink()
            deleted_count += 1
            logger.info(f"Deleted: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path.name}: {e}")
    
    print(f"\n✅ Successfully deleted {deleted_count} model files")
    logger.info(f"Cleared {deleted_count} stale model files from {model_dir}")
    
    return deleted_count


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear stale ML model files")
    parser.add_argument(
        '--force', 
        action='store_true', 
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    deleted = clear_stale_models(confirm=not args.force)
    
    if deleted > 0:
        print("\n📝 Next steps:")
        print("  1. New models will be trained automatically on next prediction")
        print("  2. Run predictions to verify accuracy:")
        print("     python3 -c \"from services.ml_service import predict_next_days; predict_next_days('AAPL', force_retrain=True)\"")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
