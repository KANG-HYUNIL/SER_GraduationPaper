import json
import logging
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

import src.models
from src.data.dataset import RavdessDataset
from src.data.transforms import AudioPipeline
from src.utils.metrics_eval import calculate_comprehensive_metrics
from src.utils.registry import get_model_class
from src.utils.viz_curves import (
    plot_calibration_curve,
    plot_learning_curves,
    plot_roc_pr_curves,
)
from src.utils.viz_embeddings import plot_tsne_embeddings
from src.utils.viz_heatmaps import plot_confusion_matrix

logger = logging.getLogger(__name__)

EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def sanitize_experiment_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(name))
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "experiment"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_dataloaders(cfg, dataset, train_idx, val_idx):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    metrics = calculate_comprehensive_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    feature_batches = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * inputs.size(0)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            features = extract_features(model, inputs)
            feature_batches.append(features.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)
    feature_array = np.concatenate(feature_batches, axis=0)

    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob=y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "features": feature_array,
    }


def extract_features(model, inputs):
    features = model.features(inputs)

    if hasattr(model, "freq_pool") and hasattr(model, "attention_layer"):
        x_time = model.freq_pool(features).squeeze(2)
        scores = model.attention_layer(x_time)
        alpha = torch.softmax(scores, dim=2)
        return torch.sum(x_time * alpha, dim=2)

    pooled = model.pool(features) if hasattr(model, "pool") else nn.functional.adaptive_avg_pool2d(features, (1, 1))
    return torch.flatten(pooled, 1)


def ensure_artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_folds_to_run(cfg) -> int:
    total_folds = int(cfg.train.k_folds)
    folds_to_run = cfg.train.get("folds_to_run")
    if folds_to_run is None:
        return total_folds

    folds_to_run = int(folds_to_run)
    if folds_to_run < 1 or folds_to_run > total_folds:
        raise ValueError(f"train.folds_to_run must be between 1 and {total_folds}, got {folds_to_run}.")
    return folds_to_run


def save_fold_learning_curve(history, artifact_dir: Path, fold: int) -> str:
    save_path = artifact_dir / f"fold_{fold}_learning_curve.png"
    plot_learning_curves(
        {
            "loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
        },
        save_path=str(save_path),
        title=f"Fold {fold} Learning Curves",
    )
    return str(save_path)


def save_global_artifacts(result, artifact_dir: Path):
    y_true = result["global_true"]
    y_pred = result["global_pred"]
    y_prob = result["global_prob"]
    features = result["global_features"]

    confusion_path = artifact_dir / "global_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, EMOTION_NAMES, save_path=str(confusion_path), title="Global Confusion Matrix")

    calibration_path = artifact_dir / "global_calibration_curve.png"
    plot_calibration_curve(y_true, y_prob, save_path=str(calibration_path))

    roc_pr_path = artifact_dir / "global_roc_pr_curves.png"
    plot_roc_pr_curves(y_true, y_prob, EMOTION_NAMES, save_path=str(roc_pr_path))

    tsne_path = artifact_dir / "global_tsne_plot.png"
    tsne_saved = False
    try:
        plot_tsne_embeddings(features, y_true, EMOTION_NAMES, save_path=str(tsne_path))
        tsne_saved = True
    except Exception as exc:
        logger.warning("Skipping t-SNE artifact generation due to plotting error: %s", exc)

    metrics_path = artifact_dir / "summary_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(result["summary_metrics"], fp, indent=2)

    fold_metrics_path = artifact_dir / "fold_metrics.json"
    with open(fold_metrics_path, "w", encoding="utf-8") as fp:
        json.dump(result["fold_metrics"], fp, indent=2)

    artifact_paths = [
        str(confusion_path),
        str(calibration_path),
        str(roc_pr_path),
        str(metrics_path),
        str(fold_metrics_path),
    ]
    if tsne_saved:
        artifact_paths.append(str(tsne_path))
    return artifact_paths


def copy_best_model_to_root(cfg, best_model_path: str) -> str | None:
    if not cfg.train.save_best_to_root:
        return None

    import hydra.utils

    root_dir = hydra.utils.get_original_cwd()
    save_dir = Path(root_dir) / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / f"best_model_{cfg.model.name}.pt"
    shutil.copy2(best_model_path, target)
    return str(target)


def run_cross_validation_experiment(cfg, artifact_root: str | os.PathLike | None = None, trial=None):
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info("Using device: %s", device)

    processor = AudioPipeline(cfg.data)
    dataset = RavdessDataset(cfg.data, transform=processor)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check cfg.data.dataset_path.")

    artifact_dir = ensure_artifact_dir(Path(artifact_root or "artifacts"))
    weights_dir = ensure_artifact_dir(Path("weights"))

    model_class = get_model_class(cfg.model.name)
    criterion = nn.CrossEntropyLoss()

    X_dummy = np.zeros(len(dataset))
    y_dummy = np.array(dataset.labels)
    groups = np.array(dataset.actor_ids)

    total_folds = int(cfg.train.k_folds)
    folds_to_run = resolve_folds_to_run(cfg)
    if folds_to_run < total_folds:
        logger.info("Running partial cross-validation: %s/%s folds", folds_to_run, total_folds)

    splitter = GroupKFold(n_splits=total_folds)
    fold_metrics = []
    global_true = []
    global_pred = []
    global_prob = []
    global_features = []
    fold_best_paths = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_dummy, y_dummy, groups=groups), start=1):
        if fold > folds_to_run:
            break

        logger.info("Starting fold %s/%s", fold, folds_to_run)
        train_loader, val_loader = build_dataloaders(cfg, dataset, train_idx, val_idx)

        model = model_class(cfg).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        best_score = float("-inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_model_path = weights_dir / f"best_model_fold{fold}.pt"

        for epoch in range(1, cfg.train.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_output = evaluate(model, val_loader, criterion, device)
            val_metrics = val_output["metrics"]
            score = val_metrics[cfg.train.objective_metric]

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                "Fold %s Epoch %s | train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                fold,
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1_macro"],
            )

            if score > best_score:
                best_score = score
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if trial is not None:
                global_step = (fold - 1) * cfg.train.epochs + epoch
                trial.report(score, step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if patience_counter >= cfg.train.early_stopping:
                logger.info("Fold %s early stopping at epoch %s", fold, epoch)
                break

        fold_curve_path = save_fold_learning_curve(history, artifact_dir, fold)

        best_model = model_class(cfg).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        fold_output = evaluate(best_model, val_loader, criterion, device)
        fold_result = deepcopy(fold_output["metrics"])
        fold_result["fold"] = fold
        fold_result["learning_curve"] = fold_curve_path
        fold_metrics.append(fold_result)
        fold_best_paths.append(str(best_model_path))

        global_true.append(fold_output["y_true"])
        global_pred.append(fold_output["y_pred"])
        global_prob.append(fold_output["y_prob"])
        global_features.append(fold_output["features"])

    global_true = np.concatenate(global_true, axis=0)
    global_pred = np.concatenate(global_pred, axis=0)
    global_prob = np.concatenate(global_prob, axis=0)
    global_features = np.concatenate(global_features, axis=0)

    summary_metrics = calculate_comprehensive_metrics(global_true, global_pred, y_prob=global_prob)
    summary_metrics["fold_accuracy_mean"] = float(np.mean([fold["accuracy"] for fold in fold_metrics]))
    summary_metrics["fold_f1_macro_mean"] = float(np.mean([fold["f1_macro"] for fold in fold_metrics]))

    best_fold = max(fold_metrics, key=lambda item: item[cfg.train.objective_metric])
    best_model_path = fold_best_paths[best_fold["fold"] - 1]
    exported_model_path = copy_best_model_to_root(cfg, best_model_path)

    result = {
        "summary_metrics": summary_metrics,
        "fold_metrics": fold_metrics,
        "best_fold": best_fold,
        "best_model_path": best_model_path,
        "exported_model_path": exported_model_path,
        "global_true": global_true,
        "global_pred": global_pred,
        "global_prob": global_prob,
        "global_features": global_features,
    }

    artifact_paths = save_global_artifacts(result, artifact_dir)
    artifact_paths.extend(fold["learning_curve"] for fold in fold_metrics)
    result["artifact_paths"] = artifact_paths
    return result


def log_result_to_mlflow(cfg, result):
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    for key, value in result["summary_metrics"].items():
        mlflow.log_metric(key, float(value))

    for fold in result["fold_metrics"]:
        fold_id = fold["fold"]
        for metric_name in ("accuracy", "f1_macro", "uar", "war", "mcc", "kappa", "ece"):
            if metric_name in fold:
                mlflow.log_metric(f"fold_{fold_id}_{metric_name}", float(fold[metric_name]))

    for artifact_path in result["artifact_paths"]:
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)

    if result.get("best_model_path") and os.path.exists(result["best_model_path"]):
        mlflow.log_artifact(result["best_model_path"])
