from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

import src.models
from src.data.dataset import RavdessDataset, UtteranceChunkDataset, collate_utterance_chunks
from src.data.noise import parse_snr_db
from src.data.noisy_dataset import NoisyRavdessDataset
from src.data.transforms import AudioPipeline
from src.engine.losses import build_criterion
from src.engine.trainer import (
    EMOTION_NAMES,
    chunking_enabled,
    evaluate,
    get_chunking_params,
    resolve_device,
    set_seed,
)
from src.utils.registry import get_model_class
from src.utils.viz_curves import plot_calibration_curve, plot_roc_pr_curves
from src.utils.viz_heatmaps import plot_confusion_matrix

logger = logging.getLogger(__name__)


def _load_winner_config(cfg: DictConfig) -> DictConfig:
    noise_cfg = OmegaConf.create(OmegaConf.to_container(cfg.noise, resolve=True))
    path = str(noise_cfg.eval.get("resolved_config_path", "")).strip()
    if not path:
        return cfg
    saved_cfg = OmegaConf.load(to_absolute_path(path))
    merged = OmegaConf.merge(saved_cfg, {"noise": noise_cfg})
    return merged


def _condition_name(noise_type: str, snr_db) -> str:
    if parse_snr_db(snr_db) is None:
        return "clean"
    snr_text = str(snr_db).replace("-", "m").replace(".", "p")
    return f"{noise_type}_snr{snr_text}"


def _build_eval_loader(cfg: DictConfig, dataset, val_idx):
    pin_memory = torch.cuda.is_available()
    if chunking_enabled(cfg):
        chunk_frames, _, eval_hop_frames = get_chunking_params(cfg)
        eval_subset = UtteranceChunkDataset(dataset, val_idx, chunk_frames=chunk_frames, hop_frames=eval_hop_frames)
        return DataLoader(
            eval_subset,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
            pin_memory=pin_memory,
            collate_fn=collate_utterance_chunks,
        )

    eval_subset = Subset(dataset, val_idx)
    return DataLoader(
        eval_subset,
        batch_size=int(cfg.noise.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin_memory,
    )


def _load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    checkpoint_path = str(cfg.noise.eval.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("noise.eval.checkpoint_path must point to the selected clean-condition checkpoint.")
    checkpoint_path = to_absolute_path(checkpoint_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_class = get_model_class(cfg.model.name)
    model = model_class(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "noise_type",
        "snr_db",
        "accuracy",
        "f1_macro",
        "uar",
        "war",
        "mcc",
        "kappa",
        "ece",
        "loss",
        "accuracy_delta_from_clean",
        "f1_macro_delta_from_clean",
        "uar_delta_from_clean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_condition_artifacts(output, condition_dir: Path, condition_name: str) -> list[str]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = []

    confusion_path = condition_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        output["y_true"],
        output["y_pred"],
        EMOTION_NAMES,
        save_path=str(confusion_path),
        title=f"{condition_name} Confusion Matrix",
    )
    artifact_paths.append(str(confusion_path))

    calibration_path = condition_dir / "calibration_curve.png"
    plot_calibration_curve(output["y_true"], output["y_prob"], save_path=str(calibration_path))
    artifact_paths.append(str(calibration_path))

    roc_pr_path = condition_dir / "roc_pr_curves.png"
    plot_roc_pr_curves(output["y_true"], output["y_prob"], EMOTION_NAMES, save_path=str(roc_pr_path))
    artifact_paths.append(str(roc_pr_path))
    return artifact_paths


def _condition_grid(cfg: DictConfig) -> list[tuple[str, str | int | float]]:
    noise_types = list(cfg.noise.eval.noise_types)
    snr_values = list(cfg.noise.eval.snr_db)
    grid = []
    include_clean = any(parse_snr_db(snr) is None for snr in snr_values)
    if include_clean:
        grid.append(("clean", "clean"))
    for noise_type in noise_types:
        for snr in snr_values:
            if parse_snr_db(snr) is None:
                continue
            grid.append((str(noise_type), snr))
    return grid


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    cfg = _load_winner_config(raw_cfg)
    set_seed(int(cfg.train.seed))
    device = resolve_device(str(cfg.train.device))
    logger.info("Using device: %s", device)
    logger.info("Noise evaluation config:\n%s", OmegaConf.to_yaml(cfg.noise))

    cfg.data.cache_features = False
    processor = AudioPipeline(cfg.data)
    clean_dataset = RavdessDataset(cfg.data, transform=processor)
    if len(clean_dataset) == 0:
        raise RuntimeError("Dataset is empty. Check cfg.data.dataset_path.")

    x_dummy = np.zeros(len(clean_dataset))
    y_dummy = np.array(clean_dataset.labels)
    groups = np.array(clean_dataset.actor_ids)
    total_folds = int(cfg.train.k_folds)
    selected_fold = int(cfg.noise.eval.fold)
    if selected_fold < 1 or selected_fold > total_folds:
        raise ValueError(f"noise.eval.fold must be between 1 and {total_folds}, got {selected_fold}.")

    splitter = GroupKFold(n_splits=total_folds)
    folds = list(splitter.split(x_dummy, y_dummy, groups=groups))
    train_idx, val_idx = folds[selected_fold - 1]
    fold_train_labels = [clean_dataset.labels[int(idx)] for idx in train_idx]
    criterion = build_criterion(cfg, fold_train_labels, num_classes=len(EMOTION_NAMES)).to(device)
    model = _load_model(cfg, device)

    output_root = Path(str(cfg.noise.eval.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    full_results = {}
    clean_metrics = None
    artifact_budget = int(cfg.noise.eval.get("max_artifact_conditions", 12))
    save_artifacts = bool(cfg.noise.eval.get("save_condition_artifacts", True))
    generation_cfg = cfg.noise.get("generation", {})

    for condition_index, (noise_type, snr_db) in enumerate(_condition_grid(cfg), start=1):
        condition_name = _condition_name(noise_type, snr_db)
        logger.info("Evaluating condition %s/%s: %s", condition_index, len(_condition_grid(cfg)), condition_name)
        if parse_snr_db(snr_db) is None:
            condition_dataset = clean_dataset
        else:
            condition_dataset = NoisyRavdessDataset(
                cfg.data,
                transform=processor,
                noise_type=noise_type,
                snr_db=snr_db,
                seed=int(generation_cfg.get("seed", 42)),
                babble_speakers=int(generation_cfg.get("babble_speakers", 4)),
                cafe_transient_count=int(generation_cfg.get("cafe_transient_count", 6)),
            )
        loader = _build_eval_loader(cfg, condition_dataset, val_idx)
        output = evaluate(model, loader, criterion, device, cfg)
        metrics = output["metrics"]
        if parse_snr_db(snr_db) is None:
            clean_metrics = metrics

        row = {
            "condition": condition_name,
            "noise_type": noise_type,
            "snr_db": snr_db,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "f1_macro": float(metrics.get("f1_macro", 0.0)),
            "uar": float(metrics.get("uar", 0.0)),
            "war": float(metrics.get("war", 0.0)),
            "mcc": float(metrics.get("mcc", 0.0)),
            "kappa": float(metrics.get("kappa", 0.0)),
            "ece": float(metrics.get("ece", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy_delta_from_clean": 0.0,
            "f1_macro_delta_from_clean": 0.0,
            "uar_delta_from_clean": 0.0,
        }
        if clean_metrics is not None:
            row["accuracy_delta_from_clean"] = row["accuracy"] - float(clean_metrics.get("accuracy", 0.0))
            row["f1_macro_delta_from_clean"] = row["f1_macro"] - float(clean_metrics.get("f1_macro", 0.0))
            row["uar_delta_from_clean"] = row["uar"] - float(clean_metrics.get("uar", 0.0))

        artifact_paths = []
        condition_dir = output_root / condition_name
        _write_json(condition_dir / "metrics.json", row)
        if save_artifacts and condition_index <= artifact_budget:
            artifact_paths = _save_condition_artifacts(output, condition_dir, condition_name)
        row["artifact_paths"] = artifact_paths
        rows.append(row)
        full_results[condition_name] = row

    if clean_metrics is not None:
        for row in rows:
            row["accuracy_delta_from_clean"] = row["accuracy"] - float(clean_metrics.get("accuracy", 0.0))
            row["f1_macro_delta_from_clean"] = row["f1_macro"] - float(clean_metrics.get("f1_macro", 0.0))
            row["uar_delta_from_clean"] = row["uar"] - float(clean_metrics.get("uar", 0.0))

    _write_summary_csv(output_root / "noise_summary.csv", rows)
    _write_json(output_root / "noise_summary.json", {"conditions": rows})
    _write_json(output_root / "resolved_noise_eval_config.json", OmegaConf.to_container(cfg, resolve=True))

    logger.info("Noise robustness summary:")
    for row in rows:
        logger.info(
            "%s | acc=%.4f f1=%.4f uar=%.4f d_acc=%+.4f",
            row["condition"],
            row["accuracy"],
            row["f1_macro"],
            row["uar"],
            row["accuracy_delta_from_clean"],
        )


if __name__ == "__main__":
    main()
