import json
import logging
import os
from pathlib import Path

import hydra
import hydra.utils
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf, open_dict

from src.engine.trainer import run_cross_validation_experiment, sanitize_experiment_name
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


def suggest_monotonic_hidden_dims(trial, choices, min_blocks, max_blocks):
    num_blocks = trial.suggest_int("cnn_num_blocks", min_blocks, max_blocks)
    hidden_dims = []
    min_choice_index = 0

    for block_idx in range(num_blocks):
        choice = trial.suggest_categorical(
            f"cnn_hidden_dim_{block_idx + 1}",
            choices[min_choice_index:],
        )
        hidden_dims.append(choice)
        min_choice_index = choices.index(choice)

    return hidden_dims


def suggest_logmel_params(trial, cfg):
    space = cfg.optuna.search_space.logmel
    sample_rate = cfg.data.sample_rate

    n_fft = trial.suggest_categorical("logmel_n_fft", list(space.n_fft_choices))
    hop_length = trial.suggest_categorical("logmel_hop_length", [v for v in space.hop_length_choices if v < n_fft])
    n_mels = trial.suggest_categorical("logmel_n_mels", list(space.n_mels_choices))
    duration = trial.suggest_categorical("logmel_duration", list(space.duration_choices))
    resize_height = trial.suggest_categorical("logmel_resize_height", list(space.resize_height_choices))
    resize_width = trial.suggest_categorical("logmel_resize_width", list(space.resize_width_choices))
    normalize = trial.suggest_categorical("logmel_normalize", list(space.normalize_choices))

    f_max_upper = sample_rate / 2
    f_max_choices = [float(v) for v in (4000.0, 6000.0, 7000.0, 8000.0) if v <= f_max_upper]
    f_min_choices = [0.0, 20.0, 50.0]
    f_min = trial.suggest_categorical("logmel_f_min", f_min_choices)
    f_max = trial.suggest_categorical("logmel_f_max", f_max_choices)
    if f_min >= f_max:
        raise optuna.TrialPruned("Invalid mel frequency range.")

    return {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "duration": duration,
        "resize_height": resize_height,
        "resize_width": resize_width,
        "normalize": normalize,
        "f_min": f_min,
        "f_max": f_max,
    }


def validate_input_resolution(hidden_dims, resize_height, resize_width):
    downsample_factor = 2 ** len(hidden_dims)
    return resize_height >= downsample_factor and resize_width >= downsample_factor


def apply_trial_params(base_cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    cnn_space = cfg.optuna.search_space.cnn
    train_space = cfg.optuna.search_space.train

    hidden_dims = suggest_monotonic_hidden_dims(
        trial,
        list(cnn_space.channel_choices),
        int(cnn_space.min_blocks),
        int(cnn_space.max_blocks),
    )
    dropout = trial.suggest_float("cnn_dropout", float(cnn_space.dropout_min), float(cnn_space.dropout_max))
    learning_rate = trial.suggest_float("train_learning_rate", float(train_space.lr_min), float(train_space.lr_max), log=True)
    weight_decay = trial.suggest_float("train_weight_decay", float(train_space.weight_decay_min), float(train_space.weight_decay_max), log=True)
    batch_size = trial.suggest_categorical("train_batch_size", list(train_space.batch_choices))
    logmel_params = suggest_logmel_params(trial, cfg)

    if not validate_input_resolution(hidden_dims, logmel_params["resize_height"], logmel_params["resize_width"]):
        raise optuna.TrialPruned("Input resolution too small for sampled CNN depth.")

    with open_dict(cfg):
        cfg.model.hidden_dims = hidden_dims
        cfg.model.dropout = dropout
        cfg.train.learning_rate = learning_rate
        cfg.train.weight_decay = weight_decay
        cfg.train.batch_size = batch_size
        cfg.train.save_best_to_root = False
        cfg.experiment.name = sanitize_experiment_name(f"{cfg.experiment.family}_trial_{trial.number:04d}")

        for key, value in logmel_params.items():
            cfg.data[key] = value

    return cfg


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
    storage = ensure_storage_path(cfg.optuna.storage, root_dir)
    sampler = optuna.samplers.TPESampler(seed=int(cfg.optuna.sampler_seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(cfg.optuna.pruner.warmup_steps))
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
            mlflow.log_text(OmegaConf.to_yaml(trial_cfg), f"{trial_dir.name}_config.yaml")
            result = run_cross_validation_experiment(trial_cfg, artifact_root=trial_dir / "artifacts", trial=trial)
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

        return float(result["summary_metrics"][cfg.optuna.metric])

    with mlflow.start_run(run_name=f"{cfg.optuna.study_name}_study"):
        study.optimize(objective, n_trials=int(cfg.optuna.trials), timeout=cfg.optuna.timeout, n_jobs=1)

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
        analyze_optuna_study(study, save_dir="optuna_plots")
        for artifact in Path("optuna_plots").glob("*"):
            mlflow.log_artifact(str(artifact))

        logger.info("Optuna study complete. Best trial=%s best_value=%.6f", study.best_trial.number, study.best_value)


if __name__ == "__main__":
    main()
