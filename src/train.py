import logging

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from src.engine.trainer import log_result_to_mlflow, run_cross_validation_experiment

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting training for experiment=%s", cfg.experiment.name)
    mlflow.set_experiment(f"SER_{cfg.experiment.family}")

    run_name = cfg.experiment.name
    if cfg.experiment.tag:
        run_name = f"{run_name}_{cfg.experiment.tag}"

    with mlflow.start_run(run_name=run_name):
        config_path = "resolved_config.yaml"
        with open(config_path, "w", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(config_path)
        result = run_cross_validation_experiment(cfg, artifact_root="artifacts")
        log_result_to_mlflow(cfg, result)

        logger.info("Training complete. Summary metrics: %s", result["summary_metrics"])
        print(OmegaConf.to_yaml({"summary_metrics": result["summary_metrics"], "best_fold": result["best_fold"]}))


if __name__ == "__main__":
    main()
