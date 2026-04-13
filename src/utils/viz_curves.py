import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def plot_calibration_curve(y_true, y_prob, save_path=None, title="Calibration Curve (Reliability Diagram)"):
    """
    Plots the calibration curve showing empirical accuracy vs predicted probability.
    """
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # In multiclass, we usually plot for the maximum probability or for a specific class
    # Here we simplify by treating it as confidence of the predicted class
    if y_prob.ndim > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        # 1 if correct, 0 if incorrect
        correct = (predictions == y_true).astype(int)
    else:
        confidences = y_prob
        correct = y_true
        
    fraction_of_positives, mean_predicted_value = calibration_curve(correct, confidences, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax1.set_ylabel("Fraction of positives (Accuracy)")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(title)

    ax2.hist(confidences, range=(0, 1), bins=10, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value (Confidence)")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(y_true, y_prob, classes, save_path=None):
    """
    Plots the ROC and PR curves for multi-class emotion classification using One-vs-Rest strategy.
    
    Args:
        y_true: True labels (integers 0 to n_classes-1)
        y_prob: Probability predictions (n_samples, n_classes)
        classes: List of class string names e.g., ["Anger", "Happy", "Sad", "Neutral"]
        save_path: Optional path to save the generated high-res image.
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # In case of binary classification, label_binarize returns 1 column
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    # ROC Curve Plot
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc:0.2f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # PR Curve Plot
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        average_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax2.plot(recall, precision, color=color, lw=2,
                 label=f'PR curve of class {classes[i]} (AP = {average_precision:0.2f})')
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve (PR)', fontsize=14)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(history, save_path=None, title="Model Learning Curves"):
    """
    Plots Training and Validation Accuracy/Loss curves from a Keras/PyTorch history object.
    
    Args:
        history: dictionary containing 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_path: Optional save path
    """
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss', lw=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', lw=2)
    plt.title('Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Train Accuracy', lw=2)
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Val Accuracy', lw=2)
    plt.title('Accuracy Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
