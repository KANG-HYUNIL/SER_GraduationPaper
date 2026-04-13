import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, cohen_kappa_score

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    Helps evaluate whether the model's confidence aligns with actual accuracy.
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    # Check if this is a multiclass output and we need the max probabilities
    if y_prob.ndim > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob
        predictions = y_prob >= 0.5
        
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i + 1]
        
        # Determine masks for probabilities falling into the current bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate top-tier academic metrics for Emotion Recognition classification.
    Returns Accuracy, F1 (Macro/Weighted), UAR, WAR, MCC, and Kappa. 
    If y_prob is provided, calculates ECE as well.
    """
    metrics = {}
    
    # Basic Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # UAR and WAR (Recall is practically Accuracy per class. UAR = Macro Recall, WAR = Weighted Recall)
    metrics['uar'] = recall_score(y_true, y_pred, average='macro')
    metrics['war'] = recall_score(y_true, y_pred, average='weighted')
    
    # Advanced Statistical Metrics
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Calibration ECE
    if y_prob is not None:
        metrics['ece'] = calculate_ece(y_true, y_prob)
        
    return metrics
