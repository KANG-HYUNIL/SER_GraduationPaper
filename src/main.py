import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch
from torch.utils.data import DataLoader
import logging
import os

# Import models to trigger registry
import src.models

from src.data.dataset import RavdessDataset
from src.data.transforms import AudioPipeline

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Print Config
    print(OmegaConf.to_yaml(cfg))
    logger.info("Starting SER Graduation Paper Project...")

    # 2. Setup MLFlow Area
    # (Optional: Set tracking URI if using a remote server, default is local ./mlruns)
    mlflow.set_experiment("SER_Baseline_Experiment")
    
    with mlflow.start_run():
        # Log all Hydra params
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # 3. Initialize Data Pipeline
        logger.info(f"Initializing Data Pipeline from {cfg.data.dataset_path}...")
        
        # Check if path exists (Hydra resolves cwd at runtime, so we might need absolute path fix if not found)
        if not os.path.exists(cfg.data.dataset_path):
             logger.error(f"Dataset path not found: {cfg.data.dataset_path}")
             return

        processor = AudioPipeline(cfg.data)
        dataset = RavdessDataset(cfg.data, transform=processor)

        if len(dataset) == 0:
            logger.error("No data found. Exiting.")
            return

        # 4. Initialize Model
        from src.utils.registry import get_model_class
        try:
            model_name = cfg.model.name
            logger.info(f"Initializing Model from Registry: {model_name}")
            model_class = get_model_class(model_name)
            model = model_class(cfg)
        except Exception as e:
            logger.error(f"Failed to initialize model '{model_name}': {e}")
            raise e

        # 5. Verify Data Loader & Model Forward
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        sample_batch, sample_labels = next(iter(loader))
        
        logger.info(f"Sample Batch Shape: {sample_batch.shape}") 
        
        # Test Forward Pass
        try:
            output = model(sample_batch)
            logger.info(f"Model Output Shape: {output.shape}") # Expected: [4, 8]
        except Exception as e:
            logger.error(f"Model forward failed: {e}")
            raise e
        
        # Log input shape for tracking
        mlflow.log_param("input_shape", str(sample_batch.shape))
        
        print("\n[SUCCESS] Pipeline & Model verification complete.")
        print(f"Loaded {len(dataset)} samples.")
        print(f"Model: {model_name}")
        print(f"Feature shape: {sample_batch.shape}")
        print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
