from __future__ import annotations

import csv
import hashlib
import html
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

import src.models  # noqa: F401 - importing registers all model classes.
from src.data.cross_corpus_dataset import COMMON_6CLASS_NAMES, CremaDSixClassDataset, RavdessSixClassDataset
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
    resolve_num_classes,
    set_seed,
)
from src.utils.registry import get_model_class

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAVDESS_DIR = REPO_ROOT / "src" / "$RVNS6MQ"
CREMAD_DIR = REPO_ROOT / "src" / "CREMA-D"


@dataclass(frozen=True)
class CheckpointEvalSpec:
    """Configuration for one presentation-time reproduction run.

    The checkpoint path stays relative to the repo, while the model/data/train
    settings are provided directly by each presentation script so they can be
    inspected and edited without chasing historical YAML files.
    """

    experiment_id: str
    checkpoint_path: str
    config: dict[str, Any]
    source_note: str
    output_name: str
    original_result_reference: dict[str, Any] = field(default_factory=dict)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def repo_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def build_runtime_config(config: dict[str, Any], *, cross_corpus: bool = False) -> DictConfig:
    cfg = OmegaConf.create(config)

    # The experimental parameter values live in each script. Dataset roots are
    # rebound here so the scripts work from this repository checkout.
    cfg.data.dataset_path = str(RAVDESS_DIR)
    if cross_corpus:
        cfg.cross_corpus.source.dataset_path = str(RAVDESS_DIR)
        cfg.cross_corpus.target.dataset_path = str(CREMAD_DIR)
        cfg.model.num_classes = len(COMMON_6CLASS_NAMES)

    if str(cfg.train.get("device", "auto")).lower().startswith("cuda") and not torch.cuda.is_available():
        cfg.train.device = "cpu"
    return cfg


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.train.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("Configured device is %s, but CUDA is unavailable. Falling back to CPU.", requested)
        return torch.device("cpu")
    return torch.device(requested)


def _date_output_dir(output_name: str) -> Path:
    now = datetime.now()
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in output_name).strip("_")
    output_dir = REPO_ROOT / "outputs" / now.strftime("%Y-%m-%d") / f"{now.strftime('%H-%M-%S')}_{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_as_jsonable(payload), fp, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<html><body><p>No rows.</p></body></html>", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    thead = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>
    <thead><tr>{thead}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(resolved),
    }


def _load_model(cfg: DictConfig, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_class = get_model_class(str(cfg.model.name))
    model = model_class(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _fold_one_indices(labels: list[int], actor_ids: list[int], n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    x_dummy = np.zeros(len(labels))
    y_dummy = np.asarray(labels)
    groups = np.asarray(actor_ids)
    folds = list(GroupKFold(n_splits=int(n_splits)).split(x_dummy, y_dummy, groups=groups))
    return folds[0]


def _build_eval_loader(cfg: DictConfig, dataset, indices: np.ndarray, batch_size: int | None = None) -> DataLoader:
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
        Subset(dataset, [int(idx) for idx in indices]),
        batch_size=int(batch_size or cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin_memory,
        collate_fn=_collate_eval_features,
    )


def _collate_eval_features(batch):
    """Pad variable-width Log-Mel features for non-chunked evaluation.

    Some retained experiments, especially the Pure Transformer runs, used
    resize_enabled=false. Their Log-Mel time dimension can vary by utterance,
    so the default PyTorch collate cannot stack a batch directly. Returning the
    original lengths preserves the masking path used by trainer.evaluate().
    """

    features, labels, lengths = zip(*batch)
    max_width = max(int(feature.shape[-1]) for feature in features)
    padded = []
    for feature in features:
        pad_width = max_width - int(feature.shape[-1])
        if pad_width > 0:
            feature = torch.nn.functional.pad(feature, (0, pad_width))
        padded.append(feature)
    return torch.stack(padded, dim=0), torch.stack(labels, dim=0), torch.as_tensor(lengths, dtype=torch.long)


def _prediction_rows(dataset, indices: list[int], output: dict[str, Any], class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for row_idx, dataset_index in enumerate(indices):
        true_idx = int(output["y_true"][row_idx])
        pred_idx = int(output["y_pred"][row_idx])
        probabilities = output["y_prob"][row_idx]
        row = {
            "row": row_idx,
            "dataset_index": int(dataset_index),
            "file": str(Path(dataset.files[int(dataset_index)]).name),
            "filepath": str(dataset.files[int(dataset_index)]),
            "true_label": class_names[true_idx],
            "pred_label": class_names[pred_idx],
            "correct": int(true_idx == pred_idx),
            "confidence": float(np.max(probabilities)),
        }
        for idx, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probabilities[idx])
        rows.append(row)
    return rows


def _metric_rows(metrics: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.generic)):
            rows.append({"scope": prefix or "evaluation", "metric": key, "value": float(value)})
    return rows


def _save_curve_artifacts(output: dict[str, Any], class_names: list[str], artifact_dir: Path, prefix: str) -> dict[str, str]:
    """Save presentation-safe numeric artifacts without matplotlib.

    The original training artifacts use matplotlib/seaborn PNGs. In the local
    Windows presentation environment, that plotting stack can terminate the
    process in native code, so these reproduction scripts save equivalent
    inspectable CSV/HTML artifacts instead.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "confusion_matrix_csv": str(artifact_dir / f"{prefix}_confusion_matrix.csv"),
        "confusion_matrix_html": str(artifact_dir / f"{prefix}_confusion_matrix.html"),
        "calibration_bins_csv": str(artifact_dir / f"{prefix}_calibration_bins.csv"),
        "calibration_bins_html": str(artifact_dir / f"{prefix}_calibration_bins.html"),
        "class_curve_summary_csv": str(artifact_dir / f"{prefix}_class_curve_summary.csv"),
        "class_curve_summary_html": str(artifact_dir / f"{prefix}_class_curve_summary.html"),
    }

    y_true = np.asarray(output["y_true"])
    y_pred = np.asarray(output["y_pred"])
    y_prob = np.asarray(output["y_prob"])
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_rows = []
    for true_idx, true_name in enumerate(class_names):
        total = int(cm[true_idx].sum())
        for pred_idx, pred_name in enumerate(class_names):
            count = int(cm[true_idx, pred_idx])
            cm_rows.append(
                {
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "count": count,
                    "row_normalized": float(count / total) if total else 0.0,
                }
            )

    confidences = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(int)
    calibration_rows = []
    for bin_idx in range(10):
        low = bin_idx / 10.0
        high = (bin_idx + 1) / 10.0
        if bin_idx == 9:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        calibration_rows.append(
            {
                "bin": bin_idx,
                "confidence_min": low,
                "confidence_max": high,
                "count": count,
                "mean_confidence": float(confidences[mask].mean()) if count else "",
                "accuracy": float(correct[mask].mean()) if count else "",
            }
        )

    class_rows = []
    for class_idx, class_name in enumerate(class_names):
        binary_true = (y_true == class_idx).astype(int)
        class_prob = y_prob[:, class_idx]
        has_positive = bool(binary_true.sum())
        has_negative = bool((1 - binary_true).sum())
        class_rows.append(
            {
                "class": class_name,
                "support": int(binary_true.sum()),
                "roc_auc_ovr": float(roc_auc_score(binary_true, class_prob)) if has_positive and has_negative else "",
                "average_precision_ovr": float(average_precision_score(binary_true, class_prob)) if has_positive else "",
            }
        )

    _write_csv(Path(paths["confusion_matrix_csv"]), cm_rows)
    _write_html_table(Path(paths["confusion_matrix_html"]), f"{prefix} confusion matrix", cm_rows)
    _write_csv(Path(paths["calibration_bins_csv"]), calibration_rows)
    _write_html_table(Path(paths["calibration_bins_html"]), f"{prefix} calibration bins", calibration_rows)
    _write_csv(Path(paths["class_curve_summary_csv"]), class_rows)
    _write_html_table(Path(paths["class_curve_summary_html"]), f"{prefix} class curve summary", class_rows)
    return paths


def _write_manifest(output_dir: Path, spec: CheckpointEvalSpec, cfg: DictConfig, checkpoint_path: Path, dataset_note: str) -> None:
    _write_json(
        output_dir / "manifest.json",
        {
            "experiment_id": spec.experiment_id,
            "source_note": spec.source_note,
            "purpose": "presentation reproduction: load saved checkpoint and run inference/evaluation again",
            "checkpoint": _fingerprint(checkpoint_path),
            "dataset_note": dataset_note,
            "original_result_reference": spec.original_result_reference,
            "model_name": str(cfg.model.name),
            "num_classes": resolve_num_classes(cfg),
            "output_dir": str(output_dir),
        },
    )
    with (output_dir / "runtime_config_from_script.yaml").open("w", encoding="utf-8") as fp:
        fp.write(OmegaConf.to_yaml(cfg))


def run_ravdess_fold1_checkpoint_eval(spec: CheckpointEvalSpec) -> Path:
    """Run checkpoint inference over the original RAVDESS fold-1 split."""

    cfg = build_runtime_config(spec.config)
    checkpoint_path = repo_path(spec.checkpoint_path)
    set_seed(int(cfg.train.seed))
    device = _resolve_device(cfg)

    processor = AudioPipeline(cfg.data)
    dataset = RavdessDataset(cfg.data, transform=processor)
    if len(dataset) == 0:
        raise RuntimeError(f"RAVDESS dataset is empty. Expected wav files under {RAVDESS_DIR}")

    train_idx, val_idx = _fold_one_indices(dataset.labels, dataset.actor_ids, int(cfg.train.k_folds))
    criterion = build_criterion(cfg, [dataset.labels[int(idx)] for idx in train_idx], num_classes=resolve_num_classes(cfg)).to(device)
    loader = _build_eval_loader(cfg, dataset, val_idx)
    model = _load_model(cfg, checkpoint_path, device)
    output = evaluate(model, loader, criterion, device, cfg)

    output_dir = _date_output_dir(spec.output_name)
    class_names = EMOTION_NAMES[: resolve_num_classes(cfg)]
    prediction_rows = _prediction_rows(dataset, [int(idx) for idx in val_idx], output, class_names)
    metric_rows = _metric_rows(output["metrics"], prefix="ravdess_fold1")

    _write_manifest(output_dir, spec, cfg, checkpoint_path, "RAVDESS GroupKFold fold 1 validation/test split")
    _write_json(output_dir / "summary_metrics.json", output["metrics"])
    _write_csv(output_dir / "summary_metrics.csv", metric_rows)
    _write_html_table(output_dir / "summary_metrics.html", f"{spec.experiment_id} summary metrics", metric_rows)
    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_html_table(output_dir / "predictions.html", f"{spec.experiment_id} predictions", prediction_rows)
    _write_json(output_dir / "artifact_index.json", _save_curve_artifacts(output, class_names, output_dir / "artifacts", "fold_1"))
    print(f"Saved presentation reproduction artifacts to: {output_dir}")
    return output_dir


def _condition_name(noise_type: str, snr_db: Any) -> str:
    if parse_snr_db(snr_db) is None:
        return "clean"
    return f"{noise_type}_snr{str(snr_db).replace('-', 'm').replace('.', 'p')}"


def _condition_grid(noise_types: list[str], snr_values: list[Any]) -> list[tuple[str, Any]]:
    grid: list[tuple[str, Any]] = []
    if any(parse_snr_db(snr) is None for snr in snr_values):
        grid.append(("clean", "clean"))
    for noise_type in noise_types:
        for snr in snr_values:
            if parse_snr_db(snr) is not None:
                grid.append((str(noise_type), snr))
    return grid


def run_noise_checkpoint_eval(spec: CheckpointEvalSpec, noise_types: list[str], snr_db: list[Any]) -> Path:
    """Run waveform-level noisy inference with the fixed clean checkpoint."""

    cfg = build_runtime_config(spec.config)
    cfg.data.cache_features = False
    checkpoint_path = repo_path(spec.checkpoint_path)
    set_seed(int(cfg.train.seed))
    device = _resolve_device(cfg)

    processor = AudioPipeline(cfg.data)
    clean_dataset = RavdessDataset(cfg.data, transform=processor)
    if len(clean_dataset) == 0:
        raise RuntimeError(f"RAVDESS dataset is empty. Expected wav files under {RAVDESS_DIR}")

    train_idx, val_idx = _fold_one_indices(clean_dataset.labels, clean_dataset.actor_ids, int(cfg.train.k_folds))
    criterion = build_criterion(cfg, [clean_dataset.labels[int(idx)] for idx in train_idx], num_classes=resolve_num_classes(cfg)).to(device)
    model = _load_model(cfg, checkpoint_path, device)
    output_dir = _date_output_dir(spec.output_name)
    class_names = EMOTION_NAMES[: resolve_num_classes(cfg)]

    summary_rows = []
    clean_metrics = None
    _write_manifest(output_dir, spec, cfg, checkpoint_path, "RAVDESS fold 1 split with waveform-level additive noise")

    for noise_type, snr in _condition_grid(noise_types, snr_db):
        condition = _condition_name(noise_type, snr)
        dataset = clean_dataset if parse_snr_db(snr) is None else NoisyRavdessDataset(
            cfg.data,
            transform=processor,
            noise_type=noise_type,
            snr_db=snr,
            seed=42,
            babble_speakers=4,
            cafe_transient_count=6,
        )
        loader = _build_eval_loader(cfg, dataset, val_idx, batch_size=16)
        output = evaluate(model, loader, criterion, device, cfg)
        metrics = dict(output["metrics"])
        if parse_snr_db(snr) is None:
            clean_metrics = metrics
        row = {"condition": condition, "noise_type": noise_type, "snr_db": snr, **metrics}
        if clean_metrics is not None:
            row["accuracy_delta_from_clean"] = float(metrics["accuracy"]) - float(clean_metrics["accuracy"])
            row["f1_macro_delta_from_clean"] = float(metrics["f1_macro"]) - float(clean_metrics["f1_macro"])
            row["uar_delta_from_clean"] = float(metrics["uar"]) - float(clean_metrics["uar"])
        summary_rows.append(row)

        condition_dir = output_dir / condition
        prediction_rows = _prediction_rows(dataset, [int(idx) for idx in val_idx], output, class_names)
        _write_json(condition_dir / "metrics.json", row)
        _write_csv(condition_dir / "predictions.csv", prediction_rows)
        _write_html_table(condition_dir / "predictions.html", f"{condition} predictions", prediction_rows)
        _save_curve_artifacts(output, class_names, condition_dir / "artifacts", condition)

    _write_json(output_dir / "noise_summary.json", {"conditions": summary_rows})
    _write_csv(output_dir / "noise_summary.csv", summary_rows)
    _write_html_table(output_dir / "noise_summary.html", f"{spec.experiment_id} noise summary", summary_rows)
    print(f"Saved presentation noise artifacts to: {output_dir}")
    return output_dir


def run_cross_corpus_checkpoint_eval(spec: CheckpointEvalSpec) -> Path:
    """Run source-validation and CREMA-D target inference from saved 6-class checkpoint."""

    cfg = build_runtime_config(spec.config, cross_corpus=True)
    checkpoint_path = repo_path(spec.checkpoint_path)
    set_seed(int(cfg.train.seed))
    device = _resolve_device(cfg)
    if not CREMAD_DIR.exists():
        raise FileNotFoundError(
            f"CREMA-D dataset is required for target reproduction but was not found at {CREMAD_DIR}. "
            "Restore the dataset there before running this script."
        )

    processor = AudioPipeline(cfg.data)
    source_dataset = RavdessSixClassDataset(str(RAVDESS_DIR), transform=processor, cache_features=bool(cfg.data.get("cache_features", True)))
    target_dataset = CremaDSixClassDataset(str(CREMAD_DIR), transform=processor, cache_features=bool(cfg.data.get("cache_features", True)))
    if len(source_dataset) == 0 or len(target_dataset) == 0:
        raise RuntimeError("Source or target dataset is empty. Check RAVDESS and CREMA-D dataset paths.")

    train_idx, val_idx = _fold_one_indices(source_dataset.labels, source_dataset.actor_ids, int(cfg.cross_corpus.train.source_folds))
    criterion = build_criterion(cfg, [source_dataset.labels[int(idx)] for idx in train_idx], num_classes=len(COMMON_6CLASS_NAMES)).to(device)
    model = _load_model(cfg, checkpoint_path, device)

    source_loader = _build_eval_loader(cfg, source_dataset, val_idx, batch_size=int(cfg.cross_corpus.train.target_batch_size))
    target_indices = np.arange(len(target_dataset))
    target_loader = _build_eval_loader(cfg, target_dataset, target_indices, batch_size=int(cfg.cross_corpus.train.target_batch_size))
    source_output = evaluate(model, source_loader, criterion, device, cfg)
    target_output = evaluate(model, target_loader, criterion, device, cfg)

    output_dir = _date_output_dir(spec.output_name)
    _write_manifest(output_dir, spec, cfg, checkpoint_path, "RAVDESS 6-class fold 1 source validation plus full CREMA-D 6-class target set")
    summary_rows = _metric_rows(source_output["metrics"], "source_val") + _metric_rows(target_output["metrics"], "target")
    _write_json(output_dir / "cross_corpus_summary.json", {"source_val_metrics": source_output["metrics"], "target_metrics": target_output["metrics"]})
    _write_csv(output_dir / "cross_corpus_summary.csv", summary_rows)
    _write_html_table(output_dir / "cross_corpus_summary.html", f"{spec.experiment_id} summary", summary_rows)
    _write_csv(output_dir / "source_val_predictions.csv", _prediction_rows(source_dataset, [int(idx) for idx in val_idx], source_output, COMMON_6CLASS_NAMES))
    _write_csv(output_dir / "target_predictions.csv", _prediction_rows(target_dataset, list(range(len(target_dataset))), target_output, COMMON_6CLASS_NAMES))
    _save_curve_artifacts(source_output, COMMON_6CLASS_NAMES, output_dir / "artifacts", "source_val")
    _save_curve_artifacts(target_output, COMMON_6CLASS_NAMES, output_dir / "artifacts", "target")
    print(f"Saved presentation cross-corpus artifacts to: {output_dir}")
    return output_dir
