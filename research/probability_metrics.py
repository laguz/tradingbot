"""
Probability Metrics for Model Evaluation

Provides metrics for evaluating probability predictions:
- Brier score
- Calibration curves
- Reliability diagrams
- ROC AUC
"""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from utils.logger import logger


def calculate_brier_score(y_true, y_prob):
    """
    Calculate Brier score (lower is better, range 0-1).
    
    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes.
    
    Args:
        y_true: Actual binary outcomes (0 or 1)
        y_prob: Predicted probabilities (0-1)
        
    Returns:
        Brier score (float)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Ensure binary outcomes
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    
    score = brier_score_loss(y_true, y_prob)
    return round(float(score), 4)


def calculate_calibration_data(y_true, y_prob, n_bins=10):
    """
    Calculate calibration curve data for reliability diagrams.
    
    Args:
        y_true: Actual binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Dict with calibration data
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Calculate expected calibration error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    ece = 0.0
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_count = np.sum(bin_mask)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
    
    return {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
        'ece': round(float(ece), 4),
        'n_bins': n_bins,
        'bin_counts': bin_counts
    }


def calculate_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate classification metrics at a given threshold.
    
    Args:
        y_true: Actual binary outcomes
        y_prob: Predicted probabilities  
        threshold: Decision threshold (default 0.5)
        
    Returns:
        Dict with precision, recall, F1, accuracy
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def compare_probability_models(y_true, prob_dict, model_names=None):
    """
    Compare multiple probability models using various metrics.
    
    Args:
        y_true: Actual binary outcomes
        prob_dict: Dict mapping model name to predicted probabilities
        model_names: Optional list of model names (uses dict keys if None)
        
    Returns:
        DataFrame with comparison metrics
    """
    import pandas as pd
    
    if model_names is None:
        model_names = list(prob_dict.keys())
    
    results = []
    
    for model_name in model_names:
        if model_name not in prob_dict:
            logger.warning(f"Model {model_name} not found in prob_dict")
            continue
            
        y_prob = prob_dict[model_name]
        
        # Calculate metrics
        brier = calculate_brier_score(y_true, y_prob)
        calib = calculate_calibration_data(y_true, y_prob)
        clf_metrics = calculate_classification_metrics(y_true, y_prob)
        
        results.append({
            'model': model_name,
            'brier_score': brier,
            'ece': calib['ece'],
            'roc_auc': clf_metrics['roc_auc'],
            'accuracy': clf_metrics['accuracy'],
            'precision': clf_metrics['precision'],
            'recall': clf_metrics['recall'],
            'f1_score': clf_metrics['f1_score']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('brier_score')  # Lower is better
    
    return df


def calculate_probability_bins(y_true, y_prob, n_bins=5):
    """
    Group predictions into probability bins and calculate accuracy per bin.
    
    Useful for understanding model calibration at different confidence levels.
    
    Args:
        y_true: Actual outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dict with bin analysis
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    bin_results = []
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            bin_probs = y_prob[bin_mask]
            bin_outcomes = y_true[bin_mask]
            
            bin_results.append({
                'bin': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                'count': int(bin_count),
                'mean_probability': round(float(np.mean(bin_probs)), 3),
                'actual_frequency': round(float(np.mean(bin_outcomes)), 3),
                'calibration_error': round(float(abs(np.mean(bin_probs) - np.mean(bin_outcomes))), 3)
            })
    
    return bin_results
