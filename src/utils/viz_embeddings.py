import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _select_tsne_subset(
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int = 240,
    max_features: int = 64,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= max_points:
        subset_features = features
        subset_labels = labels
    else:
        rng = np.random.default_rng(random_state)
        chosen_indices = []
        classes = np.unique(labels)
        per_class = max(1, max_points // max(1, len(classes)))

        for class_id in classes:
            class_indices = np.where(labels == class_id)[0]
            take = min(len(class_indices), per_class)
            chosen_indices.extend(rng.choice(class_indices, size=take, replace=False).tolist())

        remaining = max_points - len(chosen_indices)
        if remaining > 0:
            unused = np.setdiff1d(np.arange(len(features)), np.array(chosen_indices, dtype=int), assume_unique=False)
            if len(unused) > 0:
                extra = rng.choice(unused, size=min(remaining, len(unused)), replace=False)
                chosen_indices.extend(extra.tolist())

        chosen_indices = np.array(sorted(chosen_indices[:max_points]), dtype=int)
        subset_features = features[chosen_indices]
        subset_labels = labels[chosen_indices]

    if subset_features.shape[1] > max_features:
        variances = np.var(subset_features, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        subset_features = subset_features[:, top_indices]

    means = np.mean(subset_features, axis=0, keepdims=True)
    stds = np.std(subset_features, axis=0, keepdims=True)
    subset_features = (subset_features - means) / np.where(stds < 1e-8, 1.0, stds)
    return subset_features.astype(np.float64), subset_labels


def _pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    n_samples = x.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float64)
    for idx in range(n_samples - 1):
        diff = x[idx + 1 :] - x[idx]
        squared = np.sum(diff * diff, axis=1)
        distances[idx, idx + 1 :] = squared
        distances[idx + 1 :, idx] = squared
    return distances


def _hbeta(distances: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    probabilities = np.exp(-distances * beta)
    sum_probabilities = np.sum(probabilities)
    if sum_probabilities <= 0:
        probabilities = np.full_like(distances, 1.0 / max(1, len(distances)))
        sum_probabilities = np.sum(probabilities)

    entropy = np.log(sum_probabilities) + beta * np.sum(distances * probabilities) / sum_probabilities
    probabilities /= sum_probabilities
    return probabilities, entropy


def _joint_probabilities(x: np.ndarray, perplexity: float) -> np.ndarray:
    distances = _pairwise_squared_distances(x)
    n_samples = distances.shape[0]
    target_entropy = np.log(perplexity)
    conditional = np.zeros((n_samples, n_samples), dtype=np.float64)

    for idx in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[idx] = False
        distances_i = distances[idx, mask]

        beta = 1.0
        beta_min = None
        beta_max = None

        probabilities, entropy = _hbeta(distances_i, beta)
        diff = entropy - target_entropy

        for _ in range(50):
            if abs(diff) <= 1e-5:
                break
            if diff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max is None else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min is None else 0.5 * (beta + beta_min)

            probabilities, entropy = _hbeta(distances_i, beta)
            diff = entropy - target_entropy

        conditional[idx, mask] = probabilities

    joint = conditional + conditional.T
    joint /= np.maximum(np.sum(joint), 1e-12)
    return np.maximum(joint, 1e-12)


def _low_dim_affinities(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = y.shape[0]
    num = np.zeros((n_samples, n_samples), dtype=np.float64)
    for idx in range(n_samples - 1):
        diff = y[idx + 1 :] - y[idx]
        squared = np.sum(diff * diff, axis=1)
        inv = 1.0 / (1.0 + squared)
        num[idx, idx + 1 :] = inv
        num[idx + 1 :, idx] = inv

    q = num / np.maximum(np.sum(num), 1e-12)
    return num, np.maximum(q, 1e-12)


def _tsne_exact(
    x: np.ndarray,
    perplexity: float = 20.0,
    max_iter: int = 300,
    learning_rate: float = 120.0,
    random_state: int = 42,
) -> np.ndarray:
    n_samples = x.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples.")

    perplexity = min(float(perplexity), float(max(1, n_samples - 1)))
    p = _joint_probabilities(x, perplexity)
    p *= 4.0

    rng = np.random.default_rng(random_state)
    y = rng.normal(0.0, 1e-4, size=(n_samples, 2))
    y_incs = np.zeros_like(y)
    gains = np.ones_like(y)

    for iteration in range(max_iter):
        num, q = _low_dim_affinities(y)
        pq = (p - q) * num

        grad = np.zeros_like(y)
        for idx in range(n_samples):
            diff = y[idx] - y
            grad[idx] = 4.0 * np.sum(pq[:, idx][:, None] * diff, axis=0)

        gains = np.where(np.sign(grad) != np.sign(y_incs), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)

        momentum = 0.5 if iteration < 100 else 0.8
        y_incs = momentum * y_incs - learning_rate * gains * grad
        y += y_incs
        y -= np.mean(y, axis=0, keepdims=True)

        if iteration == 100:
            p /= 4.0

    return y


def plot_tsne_embeddings(features, labels, classes, save_path=None, title="t-SNE Feature Embeddings"):
    """
    Reduces model features to 2D using a lightweight exact t-SNE implementation
    that avoids native linear algebra paths known to crash in this environment.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if len(features) < 2:
        raise ValueError("t-SNE requires at least 2 samples.")

    subset_features, subset_labels = _select_tsne_subset(features, labels)
    features_2d = _tsne_exact(subset_features, perplexity=20.0, max_iter=300, random_state=42)

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(classes))
    for idx, class_name in enumerate(classes):
        mask = subset_labels == idx
        if not np.any(mask):
            continue
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=class_name,
            alpha=0.7,
            edgecolors="w",
            s=60,
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Emotions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
