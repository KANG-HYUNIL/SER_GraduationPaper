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
        
    # Optimization History
    fig_history = optuna.visualization.plot_optimization_history(study)
    
    # Parameter Importance
    fig_importance = optuna.visualization.plot_param_importances(study)
    
    # Contour Plot (Interaction between params)
    fig_contour = optuna.visualization.plot_contour(study)
    
    # Parallel Coordinate
    fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
    
    # Return figures so they can be shown inline in Jupyter,
    # or saved to disk if save_dir is requested (using plotly's write_image if kaleido is installed)
    if save_dir:
        try:
            fig_history.write_image(os.path.join(save_dir, "optuna_history.png"), scale=3)
            fig_importance.write_image(os.path.join(save_dir, "optuna_importance.png"), scale=3)
            fig_contour.write_image(os.path.join(save_dir, "optuna_contour.png"), scale=3)
            fig_parallel.write_image(os.path.join(save_dir, "optuna_parallel.png"), scale=3)
        except Exception as e:
            print("To save Optuna plots as PNG, you need the 'kaleido' package: pip install kaleido")
            print("Error:", e)
            
    return {
        'history': fig_history,
        'importance': fig_importance,
        'contour': fig_contour,
        'parallel': fig_parallel
    }
