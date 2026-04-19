import json
import logging
import os
import random
import shutil
import gc
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import src.models
from src.data.dataset import (
    ChunkedTrainDataset,
    RavdessDataset,
    UtteranceChunkDataset,
    collate_fixed_chunks,
    collate_utterance_chunks,
)
from src.data.transforms import AudioPipeline
from src.engine.losses import build_class_weights, build_criterion
from src.models.utterance_aggregators import aggregate_chunk_embeddings, aggregate_chunk_logits
from src.utils.metrics_eval import calculate_comprehensive_metrics
from src.utils.registry import get_model_class
from src.utils.viz_curves import (
    plot_calibration_curve,
    plot_learning_curves,
    plot_roc_pr_curves,
)
from src.utils.viz_embeddings import plot_tsne_embeddings
from src.utils.viz_heatmaps import plot_attention_maps, plot_cnn_feature_map, plot_confusion_matrix

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


def chunking_enabled(cfg) -> bool:
    return bool(cfg.data.get("chunking", {}).get("enabled", False))


def get_chunking_params(cfg) -> tuple[int, int, int]:
    chunk_cfg = cfg.data.get("chunking", {})
    chunk_frames = int(chunk_cfg.get("chunk_frames", 64))
    hop_frames = int(chunk_cfg.get("hop_frames", max(1, chunk_frames // 2)))
    eval_hop_frames = int(chunk_cfg.get("eval_hop_frames", hop_frames))
    return chunk_frames, hop_frames, eval_hop_frames


def get_aggregation_params(cfg) -> tuple[str, float]:
    chunk_cfg = cfg.data.get("chunking", {})
    mode = str(chunk_cfg.get("aggregation_mode", "mean_logit"))
    topk_ratio = float(chunk_cfg.get("topk_ratio", 0.5))
    return mode, topk_ratio


def build_dataloaders(cfg, dataset, train_idx, val_idx):
    pin_memory = torch.cuda.is_available()
    sampler_cfg = cfg.train.get("sampler", {})
    sampler_name = str(sampler_cfg.get("name", "random"))
    sampler_weight_mode = str(sampler_cfg.get("class_weight_mode", "none"))
    if chunking_enabled(cfg):
        chunk_frames, hop_frames, eval_hop_frames = get_chunking_params(cfg)
        train_subset = ChunkedTrainDataset(dataset, train_idx, chunk_frames=chunk_frames, hop_frames=hop_frames)
        val_subset = UtteranceChunkDataset(dataset, val_idx, chunk_frames=chunk_frames, hop_frames=eval_hop_frames)
        if len(train_subset) == 0:
            raise RuntimeError("ChunkedTrainDataset is empty. Reduce chunk_frames or adjust log-Mel parameters.")
        sampler = None
        shuffle = True
        if sampler_name == "weighted":
            chunk_labels = [dataset.labels[utterance_idx] for utterance_idx, _, _ in train_subset.chunk_index]
            class_weights = build_class_weights(chunk_labels, num_classes=len(EMOTION_NAMES), mode=sampler_weight_mode)
            if class_weights is not None:
                sample_weights = [float(class_weights[label]) for label in chunk_labels]
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                shuffle = False
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fixed_chunks,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_utterance_chunks,
        )
        return train_loader, val_loader

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": pin_memory,
    }
    sampler = None
    shuffle = True
    if sampler_name == "weighted":
        train_labels = [dataset.labels[int(idx)] for idx in train_idx]
        class_weights = build_class_weights(train_labels, num_classes=len(EMOTION_NAMES), mode=sampler_weight_mode)
        if class_weights is not None:
            sample_weights = [float(class_weights[label]) for label in train_labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
    train_loader = DataLoader(train_subset, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def unpack_batch(batch, device):
    if len(batch) == 3:
        inputs, labels, lengths = batch
        return inputs.to(device), labels.to(device), lengths.to(device)
    inputs, labels = batch
    return inputs.to(device), labels.to(device), None


def forward_model(model, inputs, lengths=None):
    if lengths is None:
        return model(inputs)
    try:
        return model(inputs, lengths=lengths)
    except TypeError:
        return model(inputs)


def apply_specaugment(inputs: torch.Tensor, cfg) -> torch.Tensor:
    specaug_cfg = cfg.data.get("specaugment", {})
    if not bool(specaug_cfg.get("enabled", False)):
        return inputs
    if inputs.ndim != 4:
        return inputs

    augmented = inputs.clone()
    _, _, freq_size, time_size = augmented.shape
    time_mask_count = int(specaug_cfg.get("time_mask_count", 0))
    time_mask_width = int(specaug_cfg.get("time_mask_width", 0))
    freq_mask_count = int(specaug_cfg.get("freq_mask_count", 0))
    freq_mask_width = int(specaug_cfg.get("freq_mask_width", 0))

    for sample_idx in range(augmented.size(0)):
        for _ in range(freq_mask_count):
            max_width = min(freq_mask_width, freq_size)
            if max_width <= 0:
                continue
            width = random.randint(0, max_width)
            if width <= 0:
                continue
            start = random.randint(0, freq_size - width)
            augmented[sample_idx, :, start : start + width, :] = 0.0

        for _ in range(time_mask_count):
            max_width = min(time_mask_width, time_size)
            if max_width <= 0:
                continue
            width = random.randint(0, max_width)
            if width <= 0:
                continue
            start = random.randint(0, time_size - width)
            augmented[sample_idx, :, :, start : start + width] = 0.0

    return augmented


def train_one_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        inputs, labels, lengths = unpack_batch(batch, device)
        inputs = apply_specaugment(inputs, cfg)

        optimizer.zero_grad()
        logits = forward_model(model, inputs, lengths)
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


def extract_features(model, inputs, lengths=None):
    if hasattr(model, "get_embedding"):
        if lengths is None:
            return model.get_embedding(inputs)
        try:
            return model.get_embedding(inputs, lengths=lengths)
        except TypeError:
            return model.get_embedding(inputs)

    features = model.features(inputs)

    if hasattr(model, "freq_pool") and hasattr(model, "attention_layer"):
        x_time = model.freq_pool(features).squeeze(2)
        scores = model.attention_layer(x_time)
        alpha = torch.softmax(scores, dim=2)
        return torch.sum(x_time * alpha, dim=2)

    pooled = model.pool(features) if hasattr(model, "pool") else nn.functional.adaptive_avg_pool2d(features, (1, 1))
    return torch.flatten(pooled, 1)


def evaluate_standard(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    feature_batches = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels, lengths = unpack_batch(batch, device)

            logits = forward_model(model, inputs, lengths)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * inputs.size(0)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            features = extract_features(model, inputs, lengths)
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


def evaluate_chunked_utterances(model, loader, criterion, device, aggregation_mode: str, topk_ratio: float):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    feature_batches = []

    with torch.no_grad():
        for chunks, labels, _ in loader:
            chunks = chunks.to(device)
            label = labels.to(device)

            chunk_logits = model(chunks)
            agg_logits, weights = aggregate_chunk_logits(chunk_logits, mode=aggregation_mode, topk_ratio=topk_ratio)
            agg_logits = agg_logits.unsqueeze(0)

            loss = criterion(agg_logits, label)
            probs = torch.softmax(agg_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            chunk_embeddings = model.get_embedding(chunks)
            utterance_embedding = aggregate_chunk_embeddings(chunk_embeddings, weights).unsqueeze(0)
            feature_batches.append(utterance_embedding.cpu().numpy())

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


def evaluate(model, loader, criterion, device, cfg):
    if chunking_enabled(cfg):
        aggregation_mode, topk_ratio = get_aggregation_params(cfg)
        return evaluate_chunked_utterances(model, loader, criterion, device, aggregation_mode, topk_ratio)
    return evaluate_standard(model, loader, criterion, device)


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


def save_model_visualizations(model, loader, device, artifact_dir: Path, cfg, fold: int) -> list[str]:
    if not hasattr(model, "enable_visualization_capture") or not hasattr(model, "get_visualization_payload"):
        return []

    batch = next(iter(loader))
    if chunking_enabled(cfg):
        chunks, _, _ = batch
        inputs = chunks[:1].to(device)
        lengths = None
    else:
        inputs, _, lengths = unpack_batch(batch, device)
        inputs = inputs[:1]
        lengths = None if lengths is None else lengths[:1]

    model.enable_visualization_capture(True)
    with torch.no_grad():
        _ = forward_model(model, inputs, lengths)
    payload = model.get_visualization_payload()
    model.enable_visualization_capture(False)

    if not payload:
        return []

    attention_path = artifact_dir / f"fold_{fold}_attention_map.png"
    feature_map_path = artifact_dir / f"fold_{fold}_cnn_feature_map.png"

    spectrogram = payload.get("spectrogram")
    attention_weights = payload.get("attention_weights")
    feature_map = payload.get("cnn_feature_map")
    if feature_map is None:
        feature_map = payload.get("frequency_feature_map")

    saved_paths = []
    if spectrogram is not None and attention_weights is not None:
        plot_attention_maps(
            spectrogram.numpy(),
            attention_weights.numpy(),
            title=f"Fold {fold} Attention Map",
            save_path=str(attention_path),
        )
        saved_paths.append(str(attention_path))

    if feature_map is not None:
        plot_cnn_feature_map(
            feature_map.numpy(),
            title=f"Fold {fold} CNN Feature Map",
            save_path=str(feature_map_path),
        )
        saved_paths.append(str(feature_map_path))

    return saved_paths


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

    best_attention_path = result.get("best_attention_map_path")
    if best_attention_path and os.path.exists(best_attention_path):
        target = artifact_dir / "attention_map.png"
        shutil.copy2(best_attention_path, target)
        artifact_paths.append(str(target))

    best_feature_map_path = result.get("best_cnn_feature_map_path")
    if best_feature_map_path and os.path.exists(best_feature_map_path):
        target = artifact_dir / "cnn_feature_map.png"
        shutil.copy2(best_feature_map_path, target)
        artifact_paths.append(str(target))
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
    x_dummy = np.zeros(len(dataset))
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
    fold_visual_paths = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_dummy, y_dummy, groups=groups), start=1):
        if fold > folds_to_run:
            break

        logger.info("Starting fold %s/%s", fold, folds_to_run)
        train_loader, val_loader = build_dataloaders(cfg, dataset, train_idx, val_idx)
        fold_train_labels = [dataset.labels[int(idx)] for idx in train_idx]
        criterion = build_criterion(cfg, fold_train_labels, num_classes=len(EMOTION_NAMES)).to(device)

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
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg)
            val_output = evaluate(model, val_loader, criterion, device, cfg)
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
        fold_output = evaluate(best_model, val_loader, criterion, device, cfg)
        visual_paths = save_model_visualizations(best_model, val_loader, device, artifact_dir, cfg, fold)
        fold_result = deepcopy(fold_output["metrics"])
        fold_result["fold"] = fold
        fold_result["learning_curve"] = fold_curve_path
        for path in visual_paths:
            if path.endswith("_attention_map.png"):
                fold_result["attention_map"] = path
            if path.endswith("_cnn_feature_map.png"):
                fold_result["cnn_feature_map"] = path
        fold_metrics.append(fold_result)
        fold_best_paths.append(str(best_model_path))
        fold_visual_paths.append(
            {
                "attention_map": fold_result.get("attention_map"),
                "cnn_feature_map": fold_result.get("cnn_feature_map"),
            }
        )

        global_true.append(fold_output["y_true"])
        global_pred.append(fold_output["y_pred"])
        global_prob.append(fold_output["y_prob"])
        global_features.append(fold_output["features"])

        del best_model, model, optimizer, train_loader, val_loader, fold_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    best_visuals = fold_visual_paths[best_fold["fold"] - 1]

    result = {
        "summary_metrics": summary_metrics,
        "fold_metrics": fold_metrics,
        "best_fold": best_fold,
        "best_model_path": best_model_path,
        "exported_model_path": exported_model_path,
        "best_attention_map_path": best_visuals.get("attention_map"),
        "best_cnn_feature_map_path": best_visuals.get("cnn_feature_map"),
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
