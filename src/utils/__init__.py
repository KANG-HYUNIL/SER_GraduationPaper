"""
SER Graduation Paper Utilities
------------------------------
This package provides comprehensive academic metrics, statistical significance 
testing, and visualization tools for Speech Emotion Recognition modeling.

Usage:
    from utils import calculate_comprehensive_metrics, plot_confusion_matrix
"""

from .metrics_eval import calculate_comprehensive_metrics, calculate_ece
from .metrics_stat import perform_mcnemar_test
from .viz_curves import plot_calibration_curve, plot_roc_pr_curves, plot_learning_curves
from .viz_heatmaps import plot_confusion_matrix, plot_attention_maps
from .viz_embeddings import plot_tsne_embeddings
from .viz_optuna import analyze_optuna_study

__all__ = [
    'calculate_comprehensive_metrics',
    'calculate_ece',
    'perform_mcnemar_test',
    'plot_calibration_curve',
    'plot_roc_pr_curves',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_attention_maps',
    'plot_tsne_embeddings',
    'analyze_optuna_study'
]
