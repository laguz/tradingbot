"""
Feature Selection Module

Reduces feature redundancy using mutual information and correlation analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from utils.logger import logger


def calculate_feature_correlations(X):
    """
    Calculate correlation matrix for features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Correlation matrix
    """
    return X.corr()


def remove_correlated_features(X, threshold=0.95):
    """
    Remove highly correlated features.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold (default 0.95)
        
    Returns:
        List of features to keep
    """
    corr_matrix = calculate_feature_correlations(X)
    
    # Find pairs of highly correlated features
    to_drop = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Drop the feature with lower variance
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                
                if X[col_i].var() < X[col_j].var():
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)
    
    features_to_keep = [f for f in X.columns if f not in to_drop]
    
    logger.info(f"Removed {len(to_drop)} highly correlated features")
    logger.info(f"Kept {len(features_to_keep)} features")
    
    return features_to_keep


def select_features_by_importance(X, y, n_features=30):
    """
    Select top N features by mutual information.
    
    Args:
        X: Feature DataFrame
        y: Target (use first column for single-output)
        n_features: Number of features to select
        
    Returns:
        List of selected feature names
    """
    logger.info(f"Calculating mutual information for {len(X.columns)} features...")
    
    # Use first target column (Day 1 prediction) for feature selection
    y_single = y.iloc[:, 0] if hasattr(y, 'iloc') else y
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y_single, random_state=42)
    
    # Create DataFrame for easy sorting
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Select top N features
    selected_features = mi_df.head(n_features)['feature'].tolist()
    
    logger.info(f"Selected top {n_features} features by mutual information")
    logger.info(f"Top 5: {selected_features[:5]}")
    
    return selected_features


def auto_select_features(X, y, max_features=35, correlation_threshold=0.95):
    """
    Automatically select best features using correlation + mutual information.
    
    Args:
        X: Feature DataFrame
        y: Target DataFrame
        max_features: Maximum number of features to keep
        correlation_threshold: Threshold for correlation removal
        
    Returns:
        List of selected feature names
    """
    logger.info(f"Auto-selecting features from {len(X.columns)} candidates")
    
    # Step 1: Remove highly correlated features
    features_after_corr = remove_correlated_features(X, correlation_threshold)
    X_filtered = X[features_after_corr]
    
    logger.info(f"After correlation filtering: {len(features_after_corr)} features")
    
    # Step 2: Select top features by mutual information
    if len(features_after_corr) > max_features:
        final_features = select_features_by_importance(X_filtered, y, max_features)
    else:
        final_features = features_after_corr
    
    logger.info(f"Final feature set: {len(final_features)} features")
    
    return final_features
