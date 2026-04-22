import json
import logging
import os
import gc
import hashlib
from pathlib import Path
from typing import Any

import hydra
import hydra.utils
import mlflow
import optuna
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import GroupKFold

import src.models
from src.data.dataset import RavdessDataset
from src.data.transforms import AudioPipeline
from src.engine.losses import build_criterion
from src.engine.trainer import build_dataloaders, resolve_device, sanitize_experiment_name
from src.engine.trainer import run_cross_validation_experiment
from src.utils.registry import get_model_class
from src.utils.viz_optuna import analyze_optuna_study

logger = logging.getLogger(__name__)


def ensure_storage_path(storage_uri: str, root_dir: Path) -> str:
    if not storage_uri.startswith("sqlite:///"):
        return storage_uri
    relative_path = storage_uri.replace("sqlite:///", "", 1)
    storage_path = Path(relative_path)
    if not storage_path.is_absolute():
        storage_path = root_dir / storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path.as_posix()}"


def resolve_study_name(cfg: DictConfig) -> str:
    base_name = str(cfg.optuna.study_name)
    if not bool(cfg.optuna.get("namespace_by_search_space", True)):
        return base_name

    payload = {
        "family": str(cfg.experiment.family),
        "model_name": str(cfg.model.name),
        "search_space": OmegaConf.to_container(cfg.optuna.search_space, resolve=True),
        "trial_overrides": OmegaConf.to_container(cfg.optuna.get("trial_overrides"), resolve=True),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{base_name}_{digest}"


def validate_dataset_path(cfg: DictConfig, root_dir: Path) -> str:
    dataset_path = Path(str(cfg.data.dataset_path))
    if not dataset_path.is_absolute():
        dataset_path = root_dir / dataset_path

    wav_count = len(list(dataset_path.glob("Actor_*/*.wav")))
    if wav_count == 0:
        raise FileNotFoundError(
            "No RAVDESS wav files found. Expected pattern "
            f"'{dataset_path / 'Actor_*' / '*.wav'}'. "
            "Set data.dataset_path to the directory that directly contains Actor_01, Actor_02, ..."
        )
    return str(dataset_path)


def _cfg_get(cfg_section: Any, key: str, default=None):
    if cfg_section is None:
        return default
    if isinstance(cfg_section, DictConfig):
        return cfg_section.get(key, default)
    return cfg_section[key] if key in cfg_section else default


def _normalize_list_of_int_lists(raw_choices) -> list[list[int]]:
    normalized: list[list[int]] = []
    for choice in raw_choices or []:
        normalized.append([int(value) for value in choice])
    return normalized


def _suggest_pair_choice(trial: optuna.Trial, name: str, raw_choices, field_name: str) -> list[int]:
    choices = _normalize_list_of_int_lists(raw_choices)
    if not choices:
        raise ValueError(f"{field_name} must define at least one candidate pair.")
    labels = ["x".join(str(value) for value in choice) for choice in choices]
    selected = trial.suggest_categorical(name, labels)
    return choices[labels.index(selected)]


def _suggest_stage_spec(trial: optuna.Trial, space) -> tuple[list[int], list[int]]:
    raw_choices = _cfg_get(space, "stage_spec_choices")
    if not raw_choices:
        stage1_dim = trial.suggest_categorical("window_stage1_dim", list(space.stage1_dim_choices))
        stage2_dim = trial.suggest_categorical("window_stage2_dim", list(space.stage2_dim_choices))
        if stage2_dim < stage1_dim:
            raise optuna.TrialPruned("window stage2_dim must be >= stage1_dim.")
        num_heads = trial.suggest_categorical("window_num_heads", list(space.num_heads_choices))
        if stage1_dim % num_heads != 0 or stage2_dim % num_heads != 0:
            raise optuna.TrialPruned("Window stage dims must be divisible by num_heads.")
        return [int(stage1_dim), int(stage2_dim)], [int(num_heads), int(num_heads)]

    stage_specs = []
    labels = []
    for spec in raw_choices:
        stage_dims = [int(value) for value in spec["stage_dims"]]
        num_heads = [int(value) for value in spec["num_heads"]]
        if len(stage_dims) != 2 or len(num_heads) != 2:
            raise ValueError("Each stage_spec_choices entry must contain two stage_dims and two num_heads values.")
        if any(dim % heads != 0 for dim, heads in zip(stage_dims, num_heads)):
            raise ValueError(f"Invalid stage spec: stage_dims={stage_dims}, num_heads={num_heads}")
        stage_specs.append((stage_dims, num_heads))
        labels.append(f"{stage_dims[0]}x{stage_dims[1]}_h{num_heads[0]}x{num_heads[1]}")

    selected = trial.suggest_categorical("window_stage_spec", labels)
    return stage_specs[labels.index(selected)]


def suggest_logmel_params(trial, cfg):
    space = cfg.optuna.search_space.logmel
    if not bool(_cfg_get(space, "enabled", True)):
        return {
            "n_fft": int(cfg.data.n_fft),
            "hop_length": int(cfg.data.hop_length),
            "n_mels": int(cfg.data.n_mels),
            "normalize": bool(cfg.data.normalize),
            "f_min": float(cfg.data.f_min),
            "f_max": float(cfg.data.f_max),
        }

    sample_rate = cfg.data.sample_rate

    n_fft = trial.suggest_categorical("logmel_n_fft", list(space.n_fft_choices))
    hop_length = trial.suggest_categorical("logmel_hop_length", list(space.hop_length_choices))
    n_mels = trial.suggest_categorical("logmel_n_mels", list(space.n_mels_choices))
    normalize = trial.suggest_categorical("logmel_normalize", list(space.normalize_choices))

    if hop_length >= n_fft:
        raise optuna.TrialPruned("Hop length must be smaller than n_fft.")

    f_max_upper = sample_rate / 2
    f_max_choices = [float(v) for v in space.f_max_choices if float(v) <= f_max_upper]
    f_min_choices = [float(v) for v in space.f_min_choices]
    if not f_max_choices:
        raise optuna.TrialPruned("No valid f_max choices under Nyquist limit.")
    f_min = trial.suggest_categorical("logmel_f_min", f_min_choices)
    f_max = trial.suggest_categorical("logmel_f_max", f_max_choices)
    if f_min >= f_max:
        raise optuna.TrialPruned("Invalid mel frequency range.")

    params = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "normalize": normalize,
        "f_min": f_min,
        "f_max": f_max,
    }
    return params


def suggest_common_train_params(trial, train_space):
    learning_rate = trial.suggest_float("train_learning_rate", float(train_space.lr_min), float(train_space.lr_max), log=True)
    weight_decay = trial.suggest_float("train_weight_decay", float(train_space.weight_decay_min), float(train_space.weight_decay_max), log=True)
    batch_size = trial.suggest_categorical("train_batch_size", list(train_space.batch_choices))
    return learning_rate, weight_decay, batch_size


def suggest_chunking_params(trial, cfg):
    space = cfg.optuna.search_space.chunking
    if not bool(_cfg_get(space, "enabled", True)):
        chunk_cfg = cfg.data.get("chunking", {})
        return {
            "enabled": bool(chunk_cfg.get("enabled", True)),
            "chunk_frames": int(chunk_cfg.get("chunk_frames", 64)),
            "hop_frames": int(chunk_cfg.get("hop_frames", 32)),
            "eval_hop_frames": int(chunk_cfg.get("eval_hop_frames", chunk_cfg.get("hop_frames", 32))),
            "aggregation_mode": str(chunk_cfg.get("aggregation_mode", "mean_logit")),
            "topk_ratio": float(chunk_cfg.get("topk_ratio", 0.5)),
        }

    chunk_frames = int(trial.suggest_categorical("chunk_frames", list(space.chunk_frames_choices)))
    hop_ratio = float(trial.suggest_categorical("chunk_hop_ratio", list(space.chunk_hop_ratio_choices)))
    eval_hop_ratio = float(trial.suggest_categorical("chunk_eval_hop_ratio", list(space.eval_hop_ratio_choices)))
    aggregation_mode = trial.suggest_categorical("chunk_aggregation_mode", list(space.aggregation_mode_choices))
    topk_ratio = float(trial.suggest_categorical("chunk_topk_ratio", list(space.topk_ratio_choices)))

    hop_frames = max(1, int(round(chunk_frames * hop_ratio)))
    eval_hop_frames = max(1, int(round(chunk_frames * eval_hop_ratio)))
    if hop_frames > chunk_frames or eval_hop_frames > chunk_frames:
        raise optuna.TrialPruned("Chunk hop size cannot exceed chunk size.")

    return {
        "enabled": True,
        "chunk_frames": chunk_frames,
        "hop_frames": hop_frames,
        "eval_hop_frames": eval_hop_frames,
        "aggregation_mode": aggregation_mode,
        "topk_ratio": topk_ratio,
    }


def suggest_pure_transformer_params(trial, cfg):
    space = cfg.optuna.search_space.transformer
    patch_size = trial.suggest_categorical("transformer_patch_size", list(space.patch_size_choices))
    patch_stride = trial.suggest_categorical("transformer_patch_stride", list(space.patch_stride_choices))
    embed_dim = trial.suggest_categorical("transformer_embed_dim", list(space.embed_dim_choices))
    num_layers = trial.suggest_int("transformer_num_layers", int(space.num_layers_min), int(space.num_layers_max))
    num_heads = trial.suggest_categorical("transformer_num_heads", list(space.num_heads_choices))
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned("Transformer embed_dim must be divisible by num_heads.")
    ffn_ratio = trial.suggest_categorical("transformer_ffn_ratio", list(space.ffn_ratio_choices))
    pooling = trial.suggest_categorical("transformer_pooling", list(space.pooling_choices))
    dropout = trial.suggest_float("transformer_dropout", float(space.dropout_min), float(space.dropout_max))
    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": int(embed_dim * ffn_ratio),
        "patch_size": [int(patch_size), int(patch_size)],
        "patch_stride": [int(patch_stride), int(patch_stride)],
        "pooling": pooling,
        "dropout": dropout,
    }


def suggest_cnn_conformer_params(trial, cfg):
    space = cfg.optuna.search_space.cnn_conformer
    backbone_variant_choices = list(_cfg_get(space, "backbone_variant_choices", ["standard"]))
    backbone_variant = str(trial.suggest_categorical("conformer_backbone_variant", backbone_variant_choices))
    overfit_strategy_choices = list(_cfg_get(space, "overfit_strategy_choices", []))
    overfit_strategy = (
        str(trial.suggest_categorical("conformer_overfit_strategy", overfit_strategy_choices))
        if overfit_strategy_choices
        else "default"
    )
    stem_pair_choices = _cfg_get(space, "stem_pair_choices")
    if stem_pair_choices:
        stem_channels = _suggest_pair_choice(trial, "conformer_stem_pair", stem_pair_choices, "stem_pair_choices")
    else:
        stem_channels = [
            trial.suggest_categorical("conformer_stem_channel_1", list(space.stem_channel_choices)),
            trial.suggest_categorical("conformer_stem_channel_2", list(space.stem_channel_choices)),
        ]
        stem_channels = sorted(stem_channels)

    raw_subsampling_choices = _cfg_get(space, "subsampling_choices")
    if not raw_subsampling_choices:
        raise ValueError("cnn_conformer.subsampling_choices must define at least one candidate.")
    subsampling_labels = [str(choice["name"]) for choice in raw_subsampling_choices]
    selected_subsampling = trial.suggest_categorical("conformer_subsampling", subsampling_labels)
    subsampling_spec = raw_subsampling_choices[subsampling_labels.index(selected_subsampling)]
    stem_strides = [[int(v) for v in pair] for pair in subsampling_spec["stem_strides"]]
    lightstem_cfg = {"channels": int(_cfg_get(space, "lightstem_default_channels", stem_channels[-1])), "stride": [2, 1]}
    raw_lightstem_stride_choices = _cfg_get(space, "lightstem_stride_choices", [])
    if raw_lightstem_stride_choices:
        labels = [str(choice["name"]) for choice in raw_lightstem_stride_choices]
        selected = trial.suggest_categorical("conformer_lightstem_stride", labels)
        spec = raw_lightstem_stride_choices[labels.index(selected)]
        lightstem_cfg["stride"] = [int(v) for v in spec["stride"]]
    raw_lightstem_channel_choices = _cfg_get(space, "lightstem_channel_choices", [])
    if raw_lightstem_channel_choices:
        lightstem_cfg["channels"] = int(trial.suggest_categorical("conformer_lightstem_channels", list(raw_lightstem_channel_choices)))
    nostem_patch_cfg = {"time_patch": int(_cfg_get(space, "nostem_patch_default_time_patch", 4))}
    raw_patch_choices = _cfg_get(space, "nostem_patch_time_patch_choices", [])
    if raw_patch_choices:
        nostem_patch_cfg["time_patch"] = int(trial.suggest_categorical("conformer_nostem_time_patch", list(raw_patch_choices)))
    band_token_cfg = {"num_bands": int(_cfg_get(space, "band_token_default_num_bands", 4))}
    raw_band_choices = _cfg_get(space, "band_token_num_bands_choices", [])
    if raw_band_choices:
        band_token_cfg["num_bands"] = int(trial.suggest_categorical("conformer_band_num_bands", list(raw_band_choices)))
    sequence_shrinking_cfg = {"enabled": False, "factor": 2, "at_layers": []}
    raw_shrinking_choices = _cfg_get(space, "sequence_shrinking_choices", [])
    if raw_shrinking_choices and overfit_strategy == "tapering":
        labels = [str(choice["name"]) for choice in raw_shrinking_choices]
        selected = trial.suggest_categorical("conformer_sequence_shrinking", labels)
        spec = raw_shrinking_choices[labels.index(selected)]
        sequence_shrinking_cfg = {
            "enabled": bool(spec.get("enabled", False)),
            "factor": int(spec.get("factor", 2)),
            "at_layers": [int(v) for v in spec.get("at_layers", [])],
        }
    layer_dim_schedule = []
    raw_layer_dim_schedule_choices = _cfg_get(space, "layer_dim_schedule_choices", [])
    if raw_layer_dim_schedule_choices and overfit_strategy == "tapering":
        labels = [str(choice["name"]) for choice in raw_layer_dim_schedule_choices]
        selected = trial.suggest_categorical("conformer_layer_dim_schedule", labels)
        spec = raw_layer_dim_schedule_choices[labels.index(selected)]
        layer_dim_schedule = [int(v) for v in spec["values"]]
    layer_ffn_schedule = []
    raw_layer_ffn_schedule_choices = _cfg_get(space, "layer_ffn_schedule_choices", [])
    if raw_layer_ffn_schedule_choices and overfit_strategy == "tapering":
        labels = [str(choice["name"]) for choice in raw_layer_ffn_schedule_choices]
        selected = trial.suggest_categorical("conformer_layer_ffn_schedule", labels)
        spec = raw_layer_ffn_schedule_choices[labels.index(selected)]
        layer_ffn_schedule = [int(v) for v in spec["values"]]
    nostem_norm_variant_choices = list(_cfg_get(space, "nostem_norm_variant_choices", []))
    nostem_norm_variant = str(_cfg_get(space, "nostem_patch_default_norm_variant", "layernorm"))
    sample_norm_always = bool(_cfg_get(space, "sample_nostem_norm_variant_always", False))
    if nostem_norm_variant_choices and (overfit_strategy == "normalization" or sample_norm_always):
        nostem_norm_variant = str(
            trial.suggest_categorical("conformer_nostem_norm_variant", nostem_norm_variant_choices)
        )
    mixup_enabled = bool(_cfg_get(space, "mixup_default_enabled", False))
    mixup_alpha = float(_cfg_get(space, "mixup_default_alpha", 0.2))
    mixup_level = str(_cfg_get(space, "mixup_default_level", "spectrogram"))
    mixup_enabled_choices = _cfg_get(space, "mixup_enabled_choices", [])
    mixup_alpha_choices = list(_cfg_get(space, "mixup_alpha_choices", []))
    mixup_level_choices = list(_cfg_get(space, "mixup_level_choices", []))
    if mixup_enabled_choices:
        mixup_enabled = bool(trial.suggest_categorical("conformer_mixup_enabled", list(mixup_enabled_choices)))
    elif overfit_strategy == "mixup":
        mixup_enabled = True
    if mixup_enabled:
        if mixup_alpha_choices:
            mixup_alpha = float(trial.suggest_categorical("conformer_mixup_alpha", mixup_alpha_choices))
        if mixup_level_choices:
            mixup_level = str(trial.suggest_categorical("conformer_mixup_level", mixup_level_choices))

    speaker_adv_enabled = bool(_cfg_get(space, "speaker_adversarial_default_enabled", False))
    speaker_adv_enabled_choices = _cfg_get(space, "speaker_adversarial_enabled_choices", [])
    if speaker_adv_enabled_choices:
        speaker_adv_enabled = bool(
            trial.suggest_categorical("conformer_speaker_adv_enabled", list(speaker_adv_enabled_choices))
        )
    speaker_adv_loss_weight = float(_cfg_get(space, "speaker_adversarial_default_loss_weight", 0.1))
    speaker_adv_grl_lambda = float(_cfg_get(space, "speaker_adversarial_default_grl_lambda", 1.0))
    speaker_adv_hidden_dim = int(_cfg_get(space, "speaker_adversarial_default_hidden_dim", 128))
    speaker_adv_dropout = float(_cfg_get(space, "speaker_adversarial_default_dropout", 0.1))
    if speaker_adv_enabled:
        loss_weight_choices = list(_cfg_get(space, "speaker_adversarial_loss_weight_choices", []))
        grl_lambda_choices = list(_cfg_get(space, "speaker_adversarial_grl_lambda_choices", []))
        hidden_dim_choices = list(_cfg_get(space, "speaker_adversarial_hidden_dim_choices", []))
        dropout_choices = list(_cfg_get(space, "speaker_adversarial_dropout_choices", []))
        if loss_weight_choices:
            speaker_adv_loss_weight = float(
                trial.suggest_categorical("conformer_speaker_adv_loss_weight", loss_weight_choices)
            )
        if grl_lambda_choices:
            speaker_adv_grl_lambda = float(
                trial.suggest_categorical("conformer_speaker_adv_grl_lambda", grl_lambda_choices)
            )
        if hidden_dim_choices:
            speaker_adv_hidden_dim = int(
                trial.suggest_categorical("conformer_speaker_adv_hidden_dim", hidden_dim_choices)
            )
        if dropout_choices:
            speaker_adv_dropout = float(
                trial.suggest_categorical("conformer_speaker_adv_dropout", dropout_choices)
            )

    embed_dim = trial.suggest_categorical("conformer_embed_dim", list(space.embed_dim_choices))
    num_layers = trial.suggest_categorical("conformer_num_layers", list(space.num_layers_choices))
    num_heads = trial.suggest_categorical("conformer_num_heads", list(space.num_heads_choices))
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned("Conformer embed_dim must be divisible by num_heads.")
    ffn_ratio = trial.suggest_categorical("conformer_ffn_ratio", list(space.ffn_ratio_choices))
    conv_kernel = trial.suggest_categorical("conformer_conv_kernel", list(space.conv_kernel_choices))
    layer_fusion = trial.suggest_categorical("conformer_layer_fusion", list(space.layer_fusion_choices))
    conv_module_type = trial.suggest_categorical("conformer_conv_module_type", list(space.conv_module_type_choices))
    loss_name = trial.suggest_categorical("conformer_loss_name", list(space.loss_name_choices))
    sampler_name = trial.suggest_categorical("conformer_sampler_name", list(space.sampler_name_choices))
    class_weight_mode = trial.suggest_categorical("conformer_class_weight_mode", list(space.class_weight_mode_choices))
    focal_gamma = float(trial.suggest_categorical("conformer_focal_gamma", list(space.focal_gamma_choices)))
    pooling = trial.suggest_categorical("conformer_pooling", list(space.pooling_choices))
    stem_dropout = trial.suggest_float("conformer_stem_dropout", float(space.stem_dropout_min), float(space.stem_dropout_max))
    projector_dropout = trial.suggest_float(
        "conformer_projector_dropout",
        float(space.projector_dropout_min),
        float(space.projector_dropout_max),
    )
    input_dropout = trial.suggest_float("conformer_input_dropout", float(space.input_dropout_min), float(space.input_dropout_max))
    encoder_dropout = trial.suggest_float(
        "conformer_encoder_dropout",
        float(space.encoder_dropout_min),
        float(space.encoder_dropout_max),
    )
    classifier_dropout = trial.suggest_float(
        "conformer_classifier_dropout",
        float(space.classifier_dropout_min),
        float(space.classifier_dropout_max),
    )
    label_smoothing = float(trial.suggest_categorical("conformer_label_smoothing", list(space.label_smoothing_choices)))
    time_mask_count = int(trial.suggest_categorical("conformer_time_mask_count", list(space.time_mask_count_choices)))
    time_mask_width = int(trial.suggest_categorical("conformer_time_mask_width", list(space.time_mask_width_choices)))
    freq_mask_count = int(trial.suggest_categorical("conformer_freq_mask_count", list(space.freq_mask_count_choices)))
    freq_mask_width = int(trial.suggest_categorical("conformer_freq_mask_width", list(space.freq_mask_width_choices)))

    model_updates = {
        "backbone_variant": backbone_variant,
        "layer_dim_schedule": layer_dim_schedule,
        "layer_ffn_schedule": layer_ffn_schedule,
        "stem_channels": [int(value) for value in stem_channels],
        "stem_strides": stem_strides,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": int(embed_dim * ffn_ratio),
        "conv_kernel_size": int(conv_kernel),
        "layer_fusion": str(layer_fusion),
        "conv_module_type": str(conv_module_type),
        "pooling": pooling,
        "dropout": encoder_dropout,
        "stem_dropout": stem_dropout,
        "projector_dropout": projector_dropout,
        "input_dropout": input_dropout,
        "encoder_dropout": encoder_dropout,
        "classifier_dropout": classifier_dropout,
        "lightstem": lightstem_cfg,
        "nostem_patch": nostem_patch_cfg,
        "band_token": {
            "num_bands": band_token_cfg["num_bands"],
            "use_band_embedding": True,
        },
        "sequence_shrinking": sequence_shrinking_cfg,
    }
    model_updates["nostem_patch"]["norm_variant"] = nostem_norm_variant
    train_updates = {
        "label_smoothing": label_smoothing,
        "loss": {
            "name": str(loss_name),
            "label_smoothing": label_smoothing,
            "class_weight_mode": str(class_weight_mode),
            "focal_gamma": focal_gamma,
        },
        "mixup": {
            "enabled": bool(mixup_enabled),
            "alpha": float(mixup_alpha),
            "level": str(mixup_level),
        },
        "sampler": {
            "name": str(sampler_name),
            "class_weight_mode": str(class_weight_mode),
        },
        "speaker_adversarial": {
            "enabled": bool(speaker_adv_enabled),
            "loss_weight": float(speaker_adv_loss_weight),
            "grl_lambda": float(speaker_adv_grl_lambda),
            "hidden_dim": int(speaker_adv_hidden_dim),
            "dropout": float(speaker_adv_dropout),
        },
    }
    data_updates = {
        "specaugment": {
            "enabled": bool(time_mask_count > 0 or freq_mask_count > 0),
            "time_mask_count": time_mask_count,
            "time_mask_width": time_mask_width,
            "freq_mask_count": freq_mask_count,
            "freq_mask_width": freq_mask_width,
        }
    }
    return model_updates, train_updates, data_updates


def suggest_hierarchical_window_params(trial, cfg):
    space = cfg.optuna.search_space.hierarchical_window
    stem_pair_choices = _cfg_get(space, "stem_pair_choices")
    if stem_pair_choices:
        stem_channels = _suggest_pair_choice(trial, "window_stem_pair", stem_pair_choices, "stem_pair_choices")
    else:
        stem_1 = trial.suggest_categorical("window_stem_channel_1", list(space.stem_channel_choices))
        stem_2 = trial.suggest_categorical("window_stem_channel_2", list(space.stem_channel_choices))
        stem_channels = sorted([int(stem_1), int(stem_2)])

    stage_dims, num_heads = _suggest_stage_spec(trial, space)

    depth_pair_choices = _cfg_get(space, "depth_pair_choices")
    if depth_pair_choices:
        stage_depths = _suggest_pair_choice(trial, "window_depth_pair", depth_pair_choices, "depth_pair_choices")
    else:
        stage1_depth = trial.suggest_int("window_stage1_depth", int(space.stage1_depth_min), int(space.stage1_depth_max))
        stage2_depth = trial.suggest_int("window_stage2_depth", int(space.stage2_depth_min), int(space.stage2_depth_max))
        stage_depths = [int(stage1_depth), int(stage2_depth)]

    window_size = int(trial.suggest_categorical("window_window_size", list(space.window_size_choices)))
    ffn_ratio = trial.suggest_categorical("window_ffn_ratio", list(space.ffn_ratio_choices))
    pooling = trial.suggest_categorical("window_pooling", list(space.pooling_choices))
    dropout = trial.suggest_float("window_dropout", float(space.dropout_min), float(space.dropout_max))
    return {
        "stem_channels": [int(value) for value in stem_channels],
        "stage_dims": [int(value) for value in stage_dims],
        "stage_depths": [int(value) for value in stage_depths],
        "num_heads": [int(value) for value in num_heads],
        "window_sizes": [window_size, window_size],
        "ffn_ratio": float(ffn_ratio),
        "pooling": pooling,
        "dropout": dropout,
        "use_shifted_windows": True,
    }


def suggest_bridged_window_params(trial, cfg):
    space = cfg.optuna.search_space.bridged_window
    stem_channels = _suggest_pair_choice(trial, "bridged_stem_pair", _cfg_get(space, "stem_pair_choices"), "stem_pair_choices")
    stage_dims, num_heads = _suggest_stage_spec(trial, space)
    stage_depths = _suggest_pair_choice(trial, "bridged_depth_pair", _cfg_get(space, "depth_pair_choices"), "depth_pair_choices")

    raw_window_choices = _cfg_get(space, "window_shape_choices")
    if not raw_window_choices:
        raise ValueError("bridged_window.window_shape_choices must define at least one candidate.")
    window_pairs = []
    labels = []
    for choice in raw_window_choices:
        stage_windows = [[int(value) for value in pair] for pair in choice["stage_windows"]]
        if len(stage_windows) != 2 or any(len(pair) != 2 for pair in stage_windows):
            raise ValueError("Each bridged window shape choice must define two [freq, time] windows.")
        window_pairs.append(stage_windows)
        labels.append(f"{stage_windows[0][0]}x{stage_windows[0][1]}_{stage_windows[1][0]}x{stage_windows[1][1]}")

    selected = trial.suggest_categorical("bridged_window_shape", labels)
    window_sizes = window_pairs[labels.index(selected)]
    bridge_tokens = int(trial.suggest_categorical("bridged_bridge_tokens", list(space.bridge_token_choices)))
    ffn_ratio = float(trial.suggest_categorical("bridged_ffn_ratio", list(space.ffn_ratio_choices)))
    pooling = trial.suggest_categorical("bridged_pooling", list(space.pooling_choices))
    dropout = trial.suggest_float("bridged_dropout", float(space.dropout_min), float(space.dropout_max))

    return {
        "stem_channels": [int(value) for value in stem_channels],
        "stage_dims": [int(value) for value in stage_dims],
        "stage_depths": [int(value) for value in stage_depths],
        "num_heads": [int(value) for value in num_heads],
        "window_sizes": [[int(v) for v in pair] for pair in window_sizes],
        "bridge_tokens": bridge_tokens,
        "ffn_ratio": ffn_ratio,
        "pooling": pooling,
        "dropout": dropout,
        "use_shifted_windows": True,
    }


def apply_trial_params(base_cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if "trial_overrides" in cfg.optuna and cfg.optuna.trial_overrides:
        cfg = OmegaConf.merge(cfg, cfg.optuna.trial_overrides)

    train_space = cfg.optuna.search_space.train
    model_name = str(cfg.model.name)
    learning_rate, weight_decay, batch_size = suggest_common_train_params(trial, train_space)
    logmel_params = suggest_logmel_params(trial, cfg)
    chunking_params = suggest_chunking_params(trial, cfg)

    model_updates = {}
    train_updates = {}
    data_updates = {}
    if model_name == "pure_transformer":
        model_updates = suggest_pure_transformer_params(trial, cfg)
    elif model_name == "cnn_conformer":
        model_updates, train_updates, data_updates = suggest_cnn_conformer_params(trial, cfg)
    elif model_name == "hierarchical_window_transformer":
        model_updates = suggest_hierarchical_window_params(trial, cfg)
    elif model_name == "bridged_window_transformer":
        model_updates = suggest_bridged_window_params(trial, cfg)
    else:
        raise ValueError(f"Unsupported Optuna model family: {model_name}")

    with open_dict(cfg):
        for key, value in model_updates.items():
            cfg.model[key] = value
        for key, value in train_updates.items():
            cfg.train[key] = value
        cfg.train.learning_rate = learning_rate
        cfg.train.weight_decay = weight_decay
        cfg.train.batch_size = batch_size
        cfg.train.save_best_to_root = False
        cfg.experiment.name = sanitize_experiment_name(f"{cfg.experiment.family}_trial_{trial.number:04d}")
        cfg.data.resize_enabled = False
        cfg.data.cache_features = True

        for key, value in logmel_params.items():
            cfg.data[key] = value
        for key, value in chunking_params.items():
            cfg.data.chunking[key] = value
        for key, value in data_updates.items():
            if isinstance(value, dict):
                if key not in cfg.data or cfg.data[key] is None:
                    cfg.data[key] = {}
                for nested_key, nested_value in value.items():
                    cfg.data[key][nested_key] = nested_value
            else:
                cfg.data[key] = value

    return cfg


def preflight_trial_config(cfg: DictConfig) -> None:
    device = resolve_device(cfg.train.device)
    processor = AudioPipeline(cfg.data)
    dataset = RavdessDataset(cfg.data, transform=processor)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty during trial preflight.")

    total_folds = int(cfg.train.k_folds)
    splitter = GroupKFold(n_splits=total_folds)
    X_dummy = np.zeros(len(dataset))
    y_dummy = np.array(dataset.labels)
    groups = np.array(dataset.actor_ids)
    train_idx, val_idx = next(iter(splitter.split(X_dummy, y_dummy, groups=groups)))
    train_loader, _ = build_dataloaders(cfg, dataset, train_idx, val_idx)
    batch = next(iter(train_loader))

    if len(batch) == 3:
        inputs, labels, lengths = batch
        lengths = lengths.to(device)
    else:
        inputs, labels = batch
        lengths = None

    inputs = inputs.to(device)
    labels = labels.to(device)

    model_class = get_model_class(cfg.model.name)
    model = model_class(cfg).to(device)
    fold_train_labels = [dataset.labels[int(idx)] for idx in train_idx]
    criterion = build_criterion(cfg, fold_train_labels, num_classes=8).to(device)

    try:
        logits = model(inputs, lengths=lengths) if lengths is not None else model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
    finally:
        del model, criterion, inputs, labels, train_loader, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_trial_summary(trial_cfg, result, trial_dir: Path):
    payload = {
        "trial_params": {
            "model": OmegaConf.to_container(trial_cfg.model, resolve=True),
            "data": OmegaConf.to_container(trial_cfg.data, resolve=True),
            "train": OmegaConf.to_container(trial_cfg.train, resolve=True),
        },
        "summary_metrics": result["summary_metrics"],
        "best_fold": result["best_fold"],
        "best_model_path": result["best_model_path"],
        "exported_model_path": result["exported_model_path"],
    }
    summary_path = trial_dir / "trial_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return str(summary_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    resolved_dataset_path = validate_dataset_path(cfg, root_dir)
    resolved_study_name = resolve_study_name(cfg)
    with open_dict(cfg):
        cfg.data.dataset_path = resolved_dataset_path
        cfg.optuna.study_name = resolved_study_name
    storage = ensure_storage_path(cfg.optuna.storage, root_dir)
    sampler = optuna.samplers.TPESampler(seed=int(cfg.optuna.sampler_seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(cfg.optuna.pruner.warmup_steps))
    logger.info("Using Optuna study: %s", cfg.optuna.study_name)
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=storage,
        direction=cfg.optuna.direction,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    mlflow.set_experiment(f"SER_OPTUNA_{cfg.experiment.family}")

    def objective(trial: optuna.Trial):
        trial_cfg = apply_trial_params(cfg, trial)
        trial_dir = Path("optuna_trials") / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "resolved_config.yaml", "w", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(trial_cfg))

        run_name = trial_cfg.experiment.name
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_artifact(str(trial_dir / "resolved_config.yaml"))
            try:
                preflight_trial_config(trial_cfg)
                result = run_cross_validation_experiment(trial_cfg, artifact_root=trial_dir / "artifacts", trial=trial)
            except ValueError as exc:
                raise optuna.TrialPruned(str(exc))
            except torch.OutOfMemoryError as exc:
                trial.set_user_attr("oom", True)
                trial.set_user_attr("oom_message", str(exc))
                mlflow.log_param("oom", True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA OOM")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    trial.set_user_attr("oom", True)
                    trial.set_user_attr("oom_message", str(exc))
                    mlflow.log_param("oom", True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned("CUDA OOM")
                raise
            summary_path = build_trial_summary(trial_cfg, result, trial_dir)

            for metric_name, metric_value in result["summary_metrics"].items():
                mlflow.log_metric(metric_name, float(metric_value))
                trial.set_user_attr(metric_name, float(metric_value))

            trial.set_user_attr("best_model_path", result["best_model_path"])
            trial.set_user_attr("trial_dir", str(trial_dir))
            mlflow.log_artifact(summary_path)
            for artifact_path in result["artifact_paths"]:
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(result["summary_metrics"][cfg.optuna.metric])

    with mlflow.start_run(run_name=f"{cfg.optuna.study_name}_study"):
        target_complete_trials = int(cfg.optuna.trials)
        study.optimize(
            objective,
            n_trials=None,
            timeout=cfg.optuna.timeout,
            n_jobs=1,
            callbacks=[MaxTrialsCallback(target_complete_trials, states=(TrialState.COMPLETE,))],
        )

        complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if not complete_trials:
            raise SystemExit(
                "Optuna finished without any COMPLETE trials. "
                "Check data.dataset_path, search-space validity, and timeout settings."
            )

        best_payload = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_attrs": study.best_trial.user_attrs,
        }
        best_path = Path("optuna_best_trial.json")
        with open(best_path, "w", encoding="utf-8") as fp:
            json.dump(best_payload, fp, indent=2)

        mlflow.log_artifact(str(best_path))
        viz_cfg = cfg.optuna.get("visualization")
        analyze_optuna_study(
            study,
            save_dir="optuna_plots",
            save_html=bool(viz_cfg.get("save_html", True)) if viz_cfg else True,
            save_png=bool(viz_cfg.get("save_png", False)) if viz_cfg else False,
            png_scale=int(viz_cfg.get("png_scale", 3)) if viz_cfg else 3,
        )
        for artifact in Path("optuna_plots").glob("*"):
            mlflow.log_artifact(str(artifact))

        logger.info("Optuna study complete. Best trial=%s best_value=%.6f", study.best_trial.number, study.best_value)


if __name__ == "__main__":
    main()
