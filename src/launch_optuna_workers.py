import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
import hydra.utils
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    worker_count = int(cfg.optuna.parallel_workers)
    trials_per_worker = int(cfg.optuna.trials_per_worker)
    root_dir = Path(hydra.utils.get_original_cwd())
    storage = str(cfg.optuna.storage)

    if worker_count > 1 and os.name == "nt" and storage.startswith("sqlite:///"):
        raise SystemExit(
            "Parallel Optuna workers are disabled on Windows when using SQLite storage. "
            "Use a single worker, switch to a server-based Optuna storage backend, "
            "or activate the environment first and launch runs sequentially."
        )

    processes = []
    for worker_id in range(worker_count):
        command = [
            sys.executable,
            "-m",
            "src.optuna_search",
            f"optuna.trials={trials_per_worker}",
            f"experiment.tag=worker{worker_id}",
        ]
        log_path = root_dir / f"optuna_worker_{worker_id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        logger.info("Launching worker %s -> %s", worker_id, " ".join(command))
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=root_dir)
        processes.append((worker_id, process, log_file))

    failed = []
    for worker_id, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failed.append(worker_id)
            logger.error("Worker %s exited with code %s", worker_id, return_code)
        else:
            logger.info("Worker %s completed successfully", worker_id)

    if failed:
        raise SystemExit(f"Optuna workers failed: {failed}")


if __name__ == "__main__":
    main()
