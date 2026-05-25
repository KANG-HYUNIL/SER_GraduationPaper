from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation_eval_common import CheckpointEvalSpec, configure_logging, run_cross_corpus_checkpoint_eval


# Source: outputs/2026-04-30/18-10-12_cross_corpus_cremad6/resolved_config.yaml
# and docs/cross_corpus/2026-04-30_RAVDESS_to_CREMAD_6class.md. This script
# does not train. It reloads artifacts/fold_1/best_model.pt and evaluates the
# same source validation fold plus the full CREMA-D 6-class target set.
EXPERIMENT_CONFIG = {
    "data": {
        "dataset_path": "",
        "sample_rate": 16000,
        "duration": 3.0,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "f_min": 0.0,
        "f_max": 8000.0,
        "normalize": True,
        "resize_enabled": True,
        "resize_height": 128,
        "resize_width": 512,
        "cache_features": True,
        "specaugment": {"enabled": False, "time_mask_count": 0, "time_mask_width": 0, "freq_mask_count": 0, "freq_mask_width": 0},
        "chunking": {"enabled": False, "chunk_frames": 64, "hop_frames": 32, "eval_hop_frames": 16, "aggregation_mode": "mean_logit", "topk_ratio": 0.5},
    },
    "model": {
        "name": "cnn_conformer",
        "num_classes": 6,
        "backbone_variant": "standard",
        "stem_channels": [32, 64],
        "stem_strides": [[2, 2], [2, 2]],
        "embed_dim": 192,
        "num_heads": 4,
        "num_layers": 8,
        "ffn_dim": 768,
        "layer_dim_schedule": [],
        "layer_ffn_schedule": [],
        "conv_kernel_size": 31,
        "conv_module_type": "single",
        "multiscale_kernel_sizes": [15, 31],
        "layer_fusion": "last",
        "pooling": "attention",
        "dropout": 0.1,
        "stem_dropout": 0.1,
        "projector_dropout": 0.1,
        "input_dropout": 0.1,
        "encoder_dropout": 0.1,
        "classifier_dropout": 0.1,
        "attention_type": "relative",
        "max_relative_position": 128,
        "lightstem": {"channels": 64, "stride": [2, 1]},
        "nostem_patch": {"time_patch": 4, "norm_variant": "layernorm"},
        "band_token": {"num_bands": 4, "use_band_embedding": True},
        "sequence_shrinking": {"enabled": False, "factor": 2, "at_layers": []},
    },
    "train": {
        "seed": 42,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "k_folds": 5,
        "folds_to_run": 1,
        "early_stopping": 10,
        "device": "auto",
        "num_workers": 0,
        "objective_metric": "f1_macro",
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0, "class_weight_mode": "none", "focal_gamma": 2.0},
        "mixup": {"enabled": False, "alpha": 0.2, "level": "spectrogram"},
        "speaker_adversarial": {"enabled": False, "loss_weight": 0.1, "grl_lambda": 1.0, "hidden_dim": 128, "dropout": 0.1},
        "sampler": {"name": "random", "class_weight_mode": "none"},
    },
    "cross_corpus": {
        "enabled": True,
        "protocol": "ravdess_to_cremad_6class",
        "class_names": ["neutral", "happy", "sad", "angry", "fearful", "disgust"],
        "source": {"name": "ravdess", "dataset_path": ""},
        "target": {"name": "cremad", "dataset_path": ""},
        "train": {"source_folds": 5, "folds_to_run": 1, "fold_selection": "first", "target_batch_size": 16, "save_fold_artifacts": True},
    },
}

SPEC = CheckpointEvalSpec(
    experiment_id="cross_corpus_cremad6_fold1_reproduction",
    checkpoint_path="outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/fold_1/best_model.pt",
    config=EXPERIMENT_CONFIG,
    source_note="Re-evaluates the saved RAVDESS->CREMA-D 6-class source-only checkpoint without retraining.",
    output_name="presentation_cross_corpus_cremad6",
    original_result_reference={
        "summary": "outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/cross_corpus_summary.json",
        "document": "docs/cross_corpus/2026-04-30_RAVDESS_to_CREMAD_6class.md",
    },
)


def main() -> None:
    configure_logging()
    run_cross_corpus_checkpoint_eval(SPEC)


if __name__ == "__main__":
    main()

