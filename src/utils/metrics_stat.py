from statsmodels.stats.contingency_tables import mcnemar

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2, alpha=0.05):
    """
    Performs McNemar's test to determine if the difference in accuracy 
    between two models (e.g. CNN and Transformer) is statistically significant.
    
    Args:
        y_true: True labels
        y_pred_model1: Predictions from Model 1
        y_pred_model2: Predictions from Model 2
        alpha: Significance level (default 0.05)
        
    Returns:
        dict: Containing the test statistic, p-value, and an interpretation.
    """
    # Create the 2x2 contingency table
    # table = [[Both correct, Model 1 correct & Model 2 wrong],
    #          [Model 1 wrong & Model 2 correct, Both wrong]]
    
    m1_correct = (y_pred_model1 == y_true)
    m2_correct = (y_pred_model2 == y_true)
    
    both_correct = sum(m1_correct & m2_correct)
    m1_only_correct = sum(m1_correct & ~m2_correct)
    m2_only_correct = sum(~m1_correct & m2_correct)
    both_wrong = sum(~m1_correct & ~m2_correct)
    
    table = [[both_correct, m1_only_correct],
             [m2_only_correct, both_wrong]]
             
    # Perform test using exact binomial distribution if frequencies are small, 
    # but the statsmodels implementation handles it automatically via 'exact' arg.
    # We default exact=True which is safer for varying sample sizes.
    result = mcnemar(table, exact=True)
    
    # Check significance
    significant = result.pvalue < alpha
    interpretation = "Statistically Significant" if significant else "Not Statistically Significant"
    
    return {
        'statistic': result.statistic,
        'pvalue': result.pvalue,
        'significant': significant,
        'interpretation': interpretation,
        'table': table
    }
