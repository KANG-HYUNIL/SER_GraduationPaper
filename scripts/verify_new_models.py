import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import optuna
import torch

from src.optuna_search import apply_trial_params
from src.utils.registry import get_model_class
import src.models


def main():
    base = OmegaConf.merge(
        OmegaConf.load("src/configs/config.yaml"),
        {"data": OmegaConf.load("src/configs/data/default.yaml")},
        {"optuna": OmegaConf.load("src/configs/optuna/default.yaml")},
    )
    if "hydra" in base:
        del base["hydra"]
    base.data.dataset_path = str(Path("src/$RVNS6MQ").resolve())
    model_files = {
        "pure_transformer": "src/configs/model/pure_transformer.yaml",
        "cnn_conformer": "src/configs/model/cnn_conformer.yaml",
        "hierarchical_window_transformer": "src/configs/model/hierarchical_window_transformer.yaml",
    }

    print("FORWARD_CHECK")
    for model_name in ("pure_transformer", "cnn_conformer", "hierarchical_window_transformer"):
        cfg = OmegaConf.merge(base, {"model": OmegaConf.load(model_files[model_name])})
        cfg.data.resize_enabled = model_name == "cnn_baseline"
        model = get_model_class(model_name)(cfg)
        x = torch.randn(2, 1, int(cfg.data.resize_height), int(cfg.data.resize_width))
        if cfg.data.resize_enabled:
            logits = model(x)
            embedding = model.get_embedding(x)
        else:
            lengths = torch.tensor([x.shape[-1], x.shape[-1] - 17])
            x[1, :, :, lengths[1]:] = 0
            logits = model(x, lengths=lengths)
            embedding = model.get_embedding(x, lengths=lengths)
        print(model_name, tuple(logits.shape), tuple(embedding.shape))

    print("OPTUNA_APPLY_CHECK")
    for model_name, path in model_files.items():
        cfg = OmegaConf.merge(base, {"model": OmegaConf.load(path)})
        cfg.data.resize_enabled = False
        study = optuna.create_study(direction="maximize")
        sampled = None
        for _ in range(20):
            trial = study.ask()
            try:
                sampled = apply_trial_params(cfg, trial)
                break
            except optuna.TrialPruned:
                continue
        if sampled is None:
            raise RuntimeError(f"Failed to sample a valid trial for {model_name}.")
        print(
            model_name,
            sampled.model.name,
            sampled.train.batch_size,
            sampled.train.learning_rate,
            sampled.train.weight_decay,
        )


if __name__ == "__main__":
    main()
