from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation_eval_common import CheckpointEvalSpec, configure_logging, run_noise_checkpoint_eval


# Source: outputs/2026-04-22/.../trial_0004/resolved_config.yaml and
# docs/noise_robustness/KR_NOISE_ROBUSTNESS_EXPERIMENT_PLAN.md. The checkpoint
# is fixed; only the input waveform is perturbed before Log-Mel extraction.
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
        "cache_features": False,
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
        "layer_dim_schedule": [],
        "layer_ffn_schedule": [],
        "conv_kernel_size": 31,
        "conv_module_type": "single",
        "multiscale_kernel_sizes": [15, 31],
        "layer_fusion": "last",
        "pooling": "attention",
        "dropout": 0.1404417693698882,
        "stem_dropout": 0.11947547746402068,
        "projector_dropout": 0.1108897907718663,
        "input_dropout": 0.0879486272613669,
        "encoder_dropout": 0.1404417693698882,
        "classifier_dropout": 0.24523691427638672,
        "attention_type": "relative",
        "max_relative_position": 128,
        "lightstem": {"channels": 96, "stride": [2, 1]},
        "nostem_patch": {"time_patch": 4, "norm_variant": "layernorm"},
        "band_token": {"num_bands": 4, "use_band_embedding": True},
        "sequence_shrinking": {"enabled": False, "factor": 2, "at_layers": []},
    },
    "train": {
        "seed": 42,
        "batch_size": 12,
        "epochs": 30,
        "learning_rate": 0.00010214082973500569,
        "weight_decay": 2.2544116997360465e-05,
        "label_smoothing": 0.0,
        "k_folds": 5,
        "folds_to_run": 1,
        "early_stopping": 10,
        "device": "auto",
        "num_workers": 0,
        "objective_metric": "f1_macro",
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0, "class_weight_mode": "none", "focal_gamma": 2.0},
        "mixup": {"enabled": True, "alpha": 0.4, "level": "spectrogram"},
        "speaker_adversarial": {"enabled": False, "loss_weight": 0.1, "grl_lambda": 1.0, "hidden_dim": 128, "dropout": 0.1},
        "sampler": {"name": "random", "class_weight_mode": "none"},
    },
}

NOISE_TYPES = ["white", "pink", "babble", "cafe"]
SNR_DB = ["clean", 20, 10, 5, 0, -5]

SPEC = CheckpointEvalSpec(
    experiment_id="noise_eval_trial_0004_fold1_reproduction",
    checkpoint_path="outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt",
    config=EXPERIMENT_CONFIG,
    source_note="Reproduces the noise robustness evaluation using the fixed trial_0004 clean checkpoint.",
    output_name="presentation_noise_eval_trial_0004",
    original_result_reference={
        "noise_summary": "outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/noise_summary.csv",
        "document": "docs/noise_robustness/KR_NOISE_ROBUSTNESS_EXPERIMENT_PLAN.md",
    },
)


def main() -> None:
    configure_logging()
    run_noise_checkpoint_eval(SPEC, noise_types=NOISE_TYPES, snr_db=SNR_DB)


if __name__ == "__main__":
    main()

