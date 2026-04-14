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
        "cnn_baseline": "src/configs/model/cnn_baseline.yaml",
        "pure_transformer": "src/configs/model/pure_transformer.yaml",
        "cnn_conformer": "src/configs/model/cnn_conformer.yaml",
        "multiscale_patch_transformer": "src/configs/model/multiscale_patch_transformer.yaml",
    }

    print("FORWARD_CHECK")
    for model_name in ("pure_transformer", "cnn_conformer", "multiscale_patch_transformer"):
        cfg = OmegaConf.merge(base, {"model": OmegaConf.load(model_files[model_name])})
        model = get_model_class(model_name)(cfg)
        x = torch.randn(2, 1, int(cfg.data.resize_height), int(cfg.data.resize_width))
        logits = model(x)
        embedding = model.get_embedding(x)
        print(model_name, tuple(logits.shape), tuple(embedding.shape))

    print("OPTUNA_APPLY_CHECK")
    for model_name, path in model_files.items():
        cfg = OmegaConf.merge(base, {"model": OmegaConf.load(path)})
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
