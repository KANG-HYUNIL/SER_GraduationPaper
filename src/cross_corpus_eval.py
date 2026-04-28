from __future__ import annotations

import csv
import json
import logging
from copy import deepcopy
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

import src.models
from src.data.cross_corpus_dataset import COMMON_6CLASS_NAMES, CremaDSixClassDataset, RavdessSixClassDataset
from src.data.dataset import UtteranceChunkDataset, collate_utterance_chunks
from src.data.transforms import AudioPipeline
from src.engine.losses import build_criterion
from src.engine.trainer import (
    build_dataloaders,
    chunking_enabled,
    evaluate,
    get_chunking_params,
    resolve_device,
    sanitize_experiment_name,
    set_seed,
    train_one_epoch,
)
from src.utils.registry import get_model_class
from src.utils.viz_curves import plot_calibration_curve, plot_roc_pr_curves
from src.utils.viz_heatmaps import plot_confusion_matrix

logger = logging.getLogger(__name__)


def _dataset_cfg(cfg: DictConfig, dataset_path: str) -> DictConfig:
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.dataset_path = str(dataset_path)
    return data_cfg


def _build_source_dataset(cfg: DictConfig, processor: AudioPipeline):
    source_name = str(cfg.cross_corpus.source.name).lower()
    source_cfg = _dataset_cfg(cfg, str(cfg.cross_corpus.source.dataset_path))
    if source_name == "ravdess":
        return RavdessSixClassDataset(source_cfg.dataset_path, transform=processor, cache_features=bool(source_cfg.get("cache_features", True)))
    raise ValueError(f"Unsupported cross_corpus.source.name: {source_name}")


def _build_target_dataset(cfg: DictConfig, processor: AudioPipeline):
    target_name = str(cfg.cross_corpus.target.name).lower()
    target_cfg = _dataset_cfg(cfg, str(cfg.cross_corpus.target.dataset_path))
    if target_name in {"cremad", "crema-d", "crema_d"}:
        return CremaDSixClassDataset(target_cfg.dataset_path, transform=processor, cache_features=bool(target_cfg.get("cache_features", True)))
    raise ValueError(f"Unsupported cross_corpus.target.name: {target_name}")


def _build_eval_loader(cfg: DictConfig, dataset, indices):
    pin_memory = torch.cuda.is_available()
    if chunking_enabled(cfg):
        chunk_frames, _, eval_hop_frames = get_chunking_params(cfg)
        subset = UtteranceChunkDataset(dataset, indices, chunk_frames=chunk_frames, hop_frames=eval_hop_frames)
        return DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
            pin_memory=pin_memory,
            collate_fn=collate_utterance_chunks,
        )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=int(cfg.cross_corpus.train.target_batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin_memory,
    )


def _select_fold_indices(total_folds: int, folds_to_run: int, mode: str) -> list[int]:
    if folds_to_run < 1 or folds_to_run > total_folds:
        raise ValueError(f"cross_corpus.train.folds_to_run must be between 1 and {total_folds}, got {folds_to_run}.")
    mode = str(mode).lower()
    if mode == "first":
        return list(range(folds_to_run))
    if mode == "all":
        return list(range(total_folds))
    raise ValueError(f"Unsupported cross_corpus.train.fold_selection: {mode}")


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_eval_artifacts(result: dict, class_names: list[str], artifact_dir: Path, prefix: str) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    confusion_path = artifact_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(result["y_true"], result["y_pred"], class_names, save_path=str(confusion_path), title=f"{prefix} Confusion Matrix")
    paths.append(str(confusion_path))

    calibration_path = artifact_dir / f"{prefix}_calibration_curve.png"
    plot_calibration_curve(result["y_true"], result["y_prob"], save_path=str(calibration_path))
    paths.append(str(calibration_path))

    roc_pr_path = artifact_dir / f"{prefix}_roc_pr_curves.png"
    plot_roc_pr_curves(result["y_true"], result["y_prob"], class_names, save_path=str(roc_pr_path))
    paths.append(str(roc_pr_path))

    return paths


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not bool(cfg.cross_corpus.enabled):
        raise ValueError("cross_corpus.enabled must be true to run src.cross_corpus_eval.")

    cfg = deepcopy(cfg)
    cfg.model.num_classes = int(len(cfg.cross_corpus.class_names))
    if list(cfg.cross_corpus.class_names) != COMMON_6CLASS_NAMES:
        logger.warning("cross_corpus.class_names differs from the built-in 6-class order. Current dataset mappings assume %s.", COMMON_6CLASS_NAMES)

    set_seed(int(cfg.train.seed))
    device = resolve_device(str(cfg.train.device))
    logger.info("Using device: %s", device)
    logger.info("Cross-corpus config:\n%s", OmegaConf.to_yaml(cfg.cross_corpus))

    processor = AudioPipeline(cfg.data)
    source_dataset = _build_source_dataset(cfg, processor)
    target_dataset = _build_target_dataset(cfg, processor)
    if len(source_dataset) == 0 or len(target_dataset) == 0:
        raise RuntimeError("Source or target dataset is empty. Check cross_corpus dataset paths.")

    x_dummy = np.zeros(len(source_dataset))
    y_dummy = np.array(source_dataset.labels)
    groups = np.array(source_dataset.actor_ids)
    total_folds = int(cfg.cross_corpus.train.source_folds)
    splitter = GroupKFold(n_splits=total_folds)
    folds = list(splitter.split(x_dummy, y_dummy, groups=groups))
    fold_ids = _select_fold_indices(total_folds, int(cfg.cross_corpus.train.folds_to_run), str(cfg.cross_corpus.train.fold_selection))
    target_indices = list(range(len(target_dataset)))

    artifact_root = Path("artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open("resolved_config.yaml", "w", encoding="utf-8") as fp:
        fp.write(OmegaConf.to_yaml(cfg))

    mlflow.set_experiment(f"SER_{sanitize_experiment_name(cfg.experiment.family)}_cross_corpus")
    run_name = f"{cfg.experiment.name}_{cfg.cross_corpus.protocol}"
    if cfg.experiment.tag:
        run_name = f"{run_name}_{cfg.experiment.tag}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact("resolved_config.yaml")

        fold_rows: list[dict] = []
        target_metric_rows: list[dict] = []
        class_names = list(cfg.cross_corpus.class_names)

        for fold_position, fold_idx in enumerate(fold_ids, start=1):
            train_idx, val_idx = folds[fold_idx]
            logger.info("Cross-corpus fold %s/%s (source fold=%s)", fold_position, len(fold_ids), fold_idx + 1)

            train_loader, val_loader = build_dataloaders(cfg, source_dataset, train_idx, val_idx)
            target_loader = _build_eval_loader(cfg, target_dataset, target_indices)

            model_class = get_model_class(cfg.model.name)
            model = model_class(cfg).to(device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=float(cfg.train.learning_rate),
                weight_decay=float(cfg.train.weight_decay),
            )
            train_labels = [source_dataset.labels[int(i)] for i in train_idx]
            criterion = build_criterion(cfg, train_labels, num_classes=int(cfg.model.num_classes)).to(device)

            best_metric = float("-inf")
            best_epoch = 0
            best_state = None
            history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
            patience = int(cfg.train.early_stopping)
            wait = 0

            for epoch in range(1, int(cfg.train.epochs) + 1):
                train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg)
                val_result = evaluate(model, val_loader, criterion, device, cfg)
                val_metrics = val_result["metrics"]

                history["train_loss"].append(float(train_metrics["loss"]))
                history["train_f1"].append(float(train_metrics["f1_macro"]))
                history["val_loss"].append(float(val_metrics["loss"]))
                history["val_f1"].append(float(val_metrics["f1_macro"]))

                metric_name = str(cfg.train.objective_metric)
                current_metric = float(val_metrics.get(metric_name, val_metrics["f1_macro"]))
                logger.info(
                    "Fold %s Epoch %s | train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f",
                    fold_idx + 1,
                    epoch,
                    train_metrics["loss"],
                    train_metrics["f1_macro"],
                    val_metrics["loss"],
                    val_metrics["f1_macro"],
                )

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info("Fold %s early stopping at epoch %s", fold_idx + 1, epoch)
                        break

            if best_state is None:
                raise RuntimeError("No best model state captured during training.")

            model.load_state_dict(best_state)
            model.to(device)
            model.eval()

            fold_dir = artifact_root / f"fold_{fold_idx + 1}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, fold_dir / "best_model.pt")
            _save_json(fold_dir / "history.json", history)

            source_val_result = evaluate(model, val_loader, criterion, device, cfg)
            target_result = evaluate(model, target_loader, criterion, device, cfg)

            source_artifacts = _save_eval_artifacts(source_val_result, class_names, fold_dir, "source_val")
            target_artifacts = _save_eval_artifacts(target_result, class_names, fold_dir, "target")

            fold_row = {
                "fold": int(fold_idx + 1),
                "best_epoch": int(best_epoch),
                "source_val_accuracy": float(source_val_result["metrics"]["accuracy"]),
                "source_val_f1_macro": float(source_val_result["metrics"]["f1_macro"]),
                "source_val_uar": float(source_val_result["metrics"]["uar"]),
                "target_accuracy": float(target_result["metrics"]["accuracy"]),
                "target_f1_macro": float(target_result["metrics"]["f1_macro"]),
                "target_uar": float(target_result["metrics"]["uar"]),
            }
            fold_rows.append(fold_row)

            target_metric_rows.append(
                {
                    "fold": int(fold_idx + 1),
                    "best_epoch": int(best_epoch),
                    "source_val_metrics": source_val_result["metrics"],
                    "target_metrics": target_result["metrics"],
                    "source_artifacts": source_artifacts,
                    "target_artifacts": target_artifacts,
                }
            )

        summary = {
            "source_val_accuracy_mean": float(np.mean([row["source_val_accuracy"] for row in fold_rows])),
            "source_val_f1_macro_mean": float(np.mean([row["source_val_f1_macro"] for row in fold_rows])),
            "source_val_uar_mean": float(np.mean([row["source_val_uar"] for row in fold_rows])),
            "target_accuracy_mean": float(np.mean([row["target_accuracy"] for row in fold_rows])),
            "target_f1_macro_mean": float(np.mean([row["target_f1_macro"] for row in fold_rows])),
            "target_uar_mean": float(np.mean([row["target_uar"] for row in fold_rows])),
        }
        best_fold = max(fold_rows, key=lambda row: row["target_f1_macro"])

        result = {
            "protocol": str(cfg.cross_corpus.protocol),
            "class_names": class_names,
            "fold_metrics": target_metric_rows,
            "summary_metrics": summary,
            "best_fold": best_fold,
        }

        _save_csv(artifact_root / "cross_corpus_fold_summary.csv", fold_rows)
        _save_json(artifact_root / "cross_corpus_summary.json", result)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        for key, value in summary.items():
            mlflow.log_metric(key, float(value))
        for row in fold_rows:
            fold_id = int(row["fold"])
            for metric_name in (
                "source_val_accuracy",
                "source_val_f1_macro",
                "source_val_uar",
                "target_accuracy",
                "target_f1_macro",
                "target_uar",
            ):
                mlflow.log_metric(f"fold_{fold_id}_{metric_name}", float(row[metric_name]))
        mlflow.log_artifact(str(artifact_root / "cross_corpus_fold_summary.csv"))
        mlflow.log_artifact(str(artifact_root / "cross_corpus_summary.json"))

        logger.info("Cross-corpus complete. Summary metrics: %s", summary)
        print(OmegaConf.to_yaml({"summary_metrics": summary, "best_fold": best_fold}))


if __name__ == "__main__":
    main()
