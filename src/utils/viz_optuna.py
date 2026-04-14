import optuna

def analyze_optuna_study(study, save_dir=None):
    """
    Generate academic-quality plots for an Optuna Hyperparameter Study.
    
    Args:
        study: An optuna.study.Study object.
        save_dir: Optional directory indicating where to save HTML/PNG plots.
    """
    import os
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

    # Return figures so they can be shown inline in Jupyter,
    # or saved to disk if save_dir is requested (using plotly's write_image if kaleido is installed)
    if save_dir:
        try:
            for name, fig in figures.items():
                fig.write_html(os.path.join(save_dir, f"optuna_{name}.html"))
                fig.write_image(os.path.join(save_dir, f"optuna_{name}.png"), scale=3)
        except Exception as e:
            print("Saved Optuna HTML plots. PNG export needs the 'kaleido' package.")
            print("Error:", e)
             
    return figures
