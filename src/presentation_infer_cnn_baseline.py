from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation_eval_common import CheckpointEvalSpec, configure_logging, run_ravdess_fold1_checkpoint_eval


# Source: the retained root checkpoint under outputs/2026-04-14/.../weights.
# The documented winner is trial_0023, but that trial's checkpoint is not
# retained separately. The only retained fold-1 checkpoint was overwritten by
# a later run; its architecture matches the late trial_0078 resolved config.
EXPERIMENT_CONFIG = {
    "data": {
        "dataset_path": "",  # rebound to this repo's src/$RVNS6MQ at runtime
        "sample_rate": 16000,
        "duration": 3.0,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 160,
        "f_min": 20.0,
        "f_max": 6000.0,
        "normalize": True,
        "resize_enabled": True,
        "resize_height": 96,
        "resize_width": 512,
        "cache_features": True,
        "chunking": {"enabled": False, "chunk_frames": 64, "hop_frames": 32, "eval_hop_frames": 16, "aggregation_mode": "mean_logit", "topk_ratio": 0.5},
    },
    "model": {"name": "cnn_baseline", "hidden_dims": [32, 32, 96, 512], "dropout": 0.3830154192099634},
    "train": {
        "seed": 42,
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 0.00014344898690472126,
        "weight_decay": 3.372998410785537e-05,
        "k_folds": 5,
        "folds_to_run": 1,
        "early_stopping": 10,
        "device": "auto",
        "num_workers": 0,
        "objective_metric": "f1_macro",
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0, "class_weight_mode": "none", "focal_gamma": 2.0},
        "mixup": {"enabled": False, "alpha": 0.2, "level": "spectrogram"},
        "sampler": {"name": "random", "class_weight_mode": "none"},
    },
}

SPEC = CheckpointEvalSpec(
    experiment_id="cnn_baseline_retained_checkpoint_fold1_reproduction",
    checkpoint_path="outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/weights/best_model_fold1.pt",
    config=EXPERIMENT_CONFIG,
    source_note="Loads the retained CNN baseline fold-1 checkpoint. The documented trial_0023 checkpoint is not retained as a separate file, and the root checkpoint was overwritten by a later trial_0078-like run.",
    output_name="presentation_cnn_baseline_retained_checkpoint",
    original_result_reference={
        "matching_trial_config": "outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/optuna_trials/trial_0078/resolved_config.yaml",
        "documented_winner_summary": "outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/optuna_trials/trial_0023/trial_summary.json",
        "document": "docs/KR_MODELS_CNN_BASELINE.md",
    },
)


def main() -> None:
    configure_logging()
    run_ravdess_fold1_checkpoint_eval(SPEC)


if __name__ == "__main__":
    main()
