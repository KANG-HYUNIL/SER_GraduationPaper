import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


EXPERIMENTS = [
    ("pure_transformer", "pure_transformer"),
    ("cnn_conformer", "cnn_conformer"),
    ("hierarchical_window_transformer", "hierarchical_window_transformer"),
]


def load_base_config(root_dir: Path):
    base = OmegaConf.merge(
        OmegaConf.load(root_dir / "src" / "configs" / "config.yaml"),
        {"data": OmegaConf.load(root_dir / "src" / "configs" / "data" / "default.yaml")},
        {"optuna": OmegaConf.load(root_dir / "src" / "configs" / "optuna" / "default.yaml")},
    )
    if "hydra" in base:
        del base["hydra"]
    return base


def build_command(
    python_executable: str,
    model_name: str,
    family: str,
    trials: int,
    epochs: int,
    folds_to_run: int | None,
    device: str,
    run_prefix: str,
    extra_overrides: list[str],
) -> list[str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{run_prefix}_{model_name}"
    study_name = f"{experiment_name}_{timestamp}"
    storage_path = f"sqlite:///optuna_studies/{study_name}.db"

    command = [
        python_executable,
        "-m",
        "src.optuna_search",
        f"model={model_name}",
        f"experiment.family={family}",
        f"experiment.name={experiment_name}",
        f"optuna.study_name={study_name}",
        f"optuna.storage={storage_path}",
        f"optuna.trials={trials}",
        f"train.epochs={epochs}",
        f"train.device={device}",
        "train.num_workers=0",
    ]
    if folds_to_run is not None:
        command.append(f"train.folds_to_run={folds_to_run}")
    command.extend(extra_overrides)
    return command


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    base_cfg = load_base_config(root_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Run the three transformer-family Optuna experiments sequentially using the "
            "default Hydra config. Recommended entrypoint: python -m scripts.run_transformer_optuna_suite"
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=24,
        help="Target COMPLETE trials per experiment. Recommended default is 24 for transformer runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs per trial. Recommended default is 15 for transformer runs.",
    )
    parser.add_argument(
        "--folds-to-run",
        type=int,
        default=1,
        help="Fold limit per trial. Default is 1 for fast transformer Optuna runs.",
    )
    parser.add_argument(
        "--device",
        default=str(base_cfg.train.device),
        help="Training device override. Default comes from src/configs/config.yaml.",
    )
    parser.add_argument(
        "--run-prefix",
        default="transformer_optuna",
        help="Prefix used in experiment names.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Number of model experiments to run in parallel. Default is sequential.",
    )
    args = parser.parse_args()

    jobs = []
    for model_name, family in EXPERIMENTS:
        command = build_command(
            python_executable=sys.executable,
            model_name=model_name,
            family=family,
            trials=args.trials,
            epochs=args.epochs,
            folds_to_run=args.folds_to_run,
            device=args.device,
            run_prefix=args.run_prefix,
            extra_overrides=args.extra_override,
        )
        jobs.append((model_name, command))

    failures: list[tuple[str, int]] = []

    def run_job(job):
        model_name, command = job
        print(f"[RUN] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=root_dir)
        return model_name, completed.returncode

    if args.max_parallel <= 1:
        for job in jobs:
            model_name, return_code = run_job(job)
            if return_code != 0:
                failures.append((model_name, return_code))
                print(f"[FAIL] {model_name} exited with code {return_code}", flush=True)
            else:
                print(f"[OK] {model_name}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                model_name, return_code = future.result()
                if return_code != 0:
                    failures.append((model_name, return_code))
                    print(f"[FAIL] {model_name} exited with code {return_code}", flush=True)
                else:
                    print(f"[OK] {model_name}", flush=True)

    if failures:
        for model_name, return_code in failures:
            print(f"[SUMMARY] {model_name}: failed ({return_code})", flush=True)
        return 1

    print("[SUMMARY] all transformer Optuna experiments completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
