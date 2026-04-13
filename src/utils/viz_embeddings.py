import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne_embeddings(features, labels, classes, save_path=None, title="t-SNE Feature Embeddings"):
    """
    Reduces CNN/Transformer high-dimensional features to 2D using t-SNE 
    and plots them to show how well emotions are clustered.
    
    Args:
        features: 2D numpy array of shape (n_samples, n_features) from the layer before classifier.
        labels: 1D numpy array of true labels.
        classes: List of string names for classes.
        save_path: Optional path to save image.
    """
    # Reduce dimension to 2D
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(classes))
    for i, class_name in enumerate(classes):
        idx = (labels == i)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    c=[colors[i]], label=class_name, alpha=0.7, edgecolors='w', s=60)
        
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
