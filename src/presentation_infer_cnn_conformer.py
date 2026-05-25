from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation_eval_common import CheckpointEvalSpec, configure_logging, run_ravdess_fold1_checkpoint_eval


# Source: the retained root checkpoint under outputs/2026-04-21/.../weights.
# The documented Chapter 4 peak is trial_0003, but its fold-1 checkpoint is
# not retained separately. The retained checkpoint has time_patch=2 and a
# tapered FFN schedule, matching trial_0041 tensor shapes.
EXPERIMENT_CONFIG = {
    "data": {
        "dataset_path": "",
        "sample_rate": 16000,
        "duration": 3.0,
        "n_mels": 80,
        "n_fft": 1024,
        "hop_length": 160,
        "f_min": 0.0,
        "f_max": 6000.0,
        "normalize": True,
        "resize_enabled": False,
        "resize_height": 128,
        "resize_width": 512,
        "cache_features": True,
        "specaugment": {"enabled": False, "time_mask_count": 0, "time_mask_width": 0, "freq_mask_count": 0, "freq_mask_width": 0},
        "chunking": {"enabled": True, "chunk_frames": 48, "hop_frames": 12, "eval_hop_frames": 12, "aggregation_mode": "confidence_weighted_logit", "topk_ratio": 0.75},
    },
    "model": {
        "name": "cnn_conformer",
        "num_classes": 8,
        "backbone_variant": "nostem_patch",
        "stem_channels": [64, 96],
        "stem_strides": [[2, 1], [2, 2]],
        "embed_dim": 192,
        "num_heads": 8,
        "num_layers": 4,
        "ffn_dim": 768,
        "layer_dim_schedule": [192, 192, 192, 192],
        "layer_ffn_schedule": [768, 768, 576, 384],
        "conv_kernel_size": 31,
        "conv_module_type": "single",
        "multiscale_kernel_sizes": [15, 31],
        "layer_fusion": "last",
        "pooling": "attention",
        "dropout": 0.1686708468614357,
        "stem_dropout": 0.11992062925899447,
        "projector_dropout": 0.0910656777444058,
        "input_dropout": 0.08164799510217961,
        "encoder_dropout": 0.1686708468614357,
        "classifier_dropout": 0.20238334792936283,
        "attention_type": "relative",
        "max_relative_position": 128,
        "lightstem": {"channels": 96, "stride": [2, 1]},
        "nostem_patch": {"time_patch": 2, "norm_variant": "layernorm"},
        "band_token": {"num_bands": 4, "use_band_embedding": True},
        "sequence_shrinking": {"enabled": False, "factor": 2, "at_layers": []},
    },
    "train": {
        "seed": 42,
        "batch_size": 12,
        "epochs": 30,
        "learning_rate": 0.00011013604781464966,
        "weight_decay": 1.4757960402673304e-05,
        "label_smoothing": 0.0,
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
    experiment_id="cnn_conformer_retained_checkpoint_fold1_reproduction",
    checkpoint_path="outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt",
    config=EXPERIMENT_CONFIG,
    source_note="Loads the retained CNN-Conformer fold-1 checkpoint. The checkpoint tensor shapes match trial_0041, while the documented trial_0003 checkpoint is not retained as a separate file.",
    output_name="presentation_cnn_conformer_retained_checkpoint",
    original_result_reference={
        "matching_trial_summary": "outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0041/trial_summary.json",
        "documented_winner_summary": "outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003/trial_summary.json",
        "document": "docs/cnn_conformer/2026-04-22_overfitting_followup.md",
    },
)


def main() -> None:
    configure_logging()
    run_ravdess_fold1_checkpoint_eval(SPEC)


if __name__ == "__main__":
    main()
