from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation_eval_common import CheckpointEvalSpec, configure_logging, run_ravdess_fold1_checkpoint_eval


# Source: the retained root checkpoint under outputs/2026-04-15/.../weights.
# The documented winner is trial_0016, but that checkpoint is not retained
# separately. The retained checkpoint has cls pooling and ffn_dim=512, matching
# trial_0153 tensor shapes, so these values follow trial_0153.
EXPERIMENT_CONFIG = {
    "data": {
        "dataset_path": "",
        "sample_rate": 16000,
        "duration": 3.0,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 160,
        "f_min": 20.0,
        "f_max": 8000.0,
        "normalize": True,
        "resize_enabled": False,
        "resize_height": 128,
        "resize_width": 512,
        "cache_features": True,
        "chunking": {"enabled": False, "chunk_frames": 64, "hop_frames": 32, "eval_hop_frames": 16, "aggregation_mode": "mean_logit", "topk_ratio": 0.5},
    },
    "model": {
        "name": "pure_transformer",
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 5,
        "ffn_dim": 512,
        "patch_size": [32, 32],
        "patch_stride": [16, 16],
        "pooling": "cls",
        "dropout": 0.1000235327588693,
    },
    "train": {
        "seed": 42,
        "batch_size": 8,
        "epochs": 30,
        "learning_rate": 0.00020923287897513278,
        "weight_decay": 4.6961385360588024e-05,
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
    experiment_id="pure_transformer_retained_checkpoint_fold1_reproduction",
    checkpoint_path="outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/weights/best_model_fold1.pt",
    config=EXPERIMENT_CONFIG,
    source_note="Loads the retained pure_transformer fold-1 checkpoint. The checkpoint tensor shapes match trial_0153, while the documented trial_0016 checkpoint is not retained as a separate file.",
    output_name="presentation_pure_transformer_retained_checkpoint",
    original_result_reference={
        "matching_trial_summary": "outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/optuna_trials/trial_0153/trial_summary.json",
        "documented_winner_summary": "outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/optuna_trials/trial_0016/trial_summary.json",
        "document": "docs/KR_MODEL_PURE_TRANSFORMER.md",
    },
)


def main() -> None:
    configure_logging()
    run_ravdess_fold1_checkpoint_eval(SPEC)


if __name__ == "__main__":
    main()
