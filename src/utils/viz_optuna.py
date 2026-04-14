import logging
import os

import optuna

logger = logging.getLogger(__name__)


def analyze_optuna_study(study, save_dir=None, save_html=True, save_png=False, png_scale=3):
    """
    Generate academic-quality plots for an Optuna Hyperparameter Study.

    Args:
        study: An optuna.study.Study object.
        save_dir: Optional directory indicating where to save HTML/PNG plots.
        save_html: Whether to save Plotly HTML files.
        save_png: Whether to export PNG files via kaleido/chrome.
        png_scale: Export scale for PNG output.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    figures = {}

    def try_add(name, builder):
        try:
            figures[name] = builder()
        except Exception as exc:
            print(f"Skipped Optuna {name} plot:", exc)

    try_add("history", lambda: optuna.visualization.plot_optimization_history(study))

    if len(completed_trials) >= 2:
        try_add("importance", lambda: optuna.visualization.plot_param_importances(study))
        try_add("contour", lambda: optuna.visualization.plot_contour(study))
        try_add("parallel", lambda: optuna.visualization.plot_parallel_coordinate(study))

    if save_dir:
        for name, fig in figures.items():
            if save_html:
                fig.write_html(os.path.join(save_dir, f"optuna_{name}.html"))
            if save_png:
                try:
                    fig.write_image(os.path.join(save_dir, f"optuna_{name}.png"), scale=png_scale)
                except Exception as exc:
                    logger.warning("Skipping Optuna PNG export for %s: %s", name, exc)

    return figures
