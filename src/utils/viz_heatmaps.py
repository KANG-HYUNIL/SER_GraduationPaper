import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, title="Normalized Confusion Matrix", cmap="Blues"):
    """
    Plots a highly stylized normalized confusion matrix (heatmap) for academic papers.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    # annot_kws to set fontsize
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12}, vmin=0, vmax=1)
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Emotion', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_maps(audio_spectrogram, attention_weights, title="Attention Map overlay", save_path=None):
    """
    Overlays attention weights onto the original Log-Mel spectrogram.
    Demonstrates which time frames the model focused on (Supports Thesis argument).
    
    Args:
        audio_spectrogram: 2D numpy array containing the spectrogram.
        attention_weights: 1D or 2D numpy array representing attention focus over time/frequency.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot original spectrogram as background
    plt.imshow(audio_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Log-Mel Amplitude')
    
    # Overlay attention weights
    # Assuming attention_weights is a 1D array corresponding to time frames
    if attention_weights.ndim == 1:
        # Extend 1D to 2D to match the height of the spectrogram for overlay visual
        attention_2d = np.tile(attention_weights, (audio_spectrogram.shape[0], 1))
        # Overlay with alpha
        plt.imshow(attention_2d, aspect='auto', origin='lower', cmap='Reds', alpha=0.4)
    else:
        plt.imshow(attention_weights, aspect='auto', origin='lower', cmap='Reds', alpha=0.4)
        
    plt.title(title, fontsize=16)
    plt.ylabel('Mel frequency bands', fontsize=12)
    plt.xlabel('Time frames', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
