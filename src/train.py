import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import mlflow
import logging
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from datetime import datetime
import shutil
import hydra.utils

# Import models to trigger registry
import src.models
from src.data.dataset import RavdessDataset
from src.data.transforms import AudioPipeline
from src.utils.registry import get_model_class

logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_learning_curves(history, save_path):
    """
    학습 곡선 그리기
    Args:
        history: 학습 기록
        save_path: 저장 경로
    Returns:
        None
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_accuracy(cm, class_names, save_path):
    """
    클래스별 정확도 그리기
    Args:
        cm: 혼동 행렬
        class_names: 클래스 이름
        save_path: 저장 경로
    Returns:
        None
    """
    # CM: rows=True, cols=Pred
    # Per-class Accuracy = Diagonal / Row_Sum
    # Handle division by zero
    row_sum = cm.sum(axis=1)
    row_sum[row_sum == 0] = 1  # Prevent division by zero
    class_acc = cm.diagonal() / row_sum
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=class_acc, palette='viridis')
    plt.title('Class-wise Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate(class_acc):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(features, labels, class_names, save_path):
    """
    t-SNE 시각화
    Args:
        features: 추출된 특징 벡터 (N, D)
        labels: 정답 레이블 (N,)
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    projections = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=projections[:, 0], y=projections[:, 1],
        hue=[class_names[i] for i in labels],
        palette='viridis',
        s=50, alpha=0.7
    )
    plt.title('t-SNE Visualization of Feature Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model: The model to train.
        loader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
    Returns:
        Average loss and accuracy for the epoch.
    """

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Gradient 초기화
        outputs = model(inputs)
        loss = criterion(outputs, labels) # Loss 계산
        loss.backward() # Gradient 계산
        optimizer.step() # Gradient 업데이트
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def validate(model, loader, criterion, device):
    """
    Validate the model for one epoch.
    Args:
        model: The model to validate.
        loader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device to validate on.
    Returns:
        Average loss, accuracy, F1 score, true labels, and predicted labels for the epoch.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_labels, all_preds

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "auto" else "cpu")
    logger.info(f"Using device: {device}")
    
    # MLFlow Setup
    mlflow.set_experiment(f"SER_{cfg.model.name}")
    
    # 2. Data Pipeline
    processor = AudioPipeline(cfg.data)
    dataset = RavdessDataset(cfg.data, transform=processor)
    
    if len(dataset) == 0:
        logger.error("No dataset found.")
        return

    # 3. Group K-Fold Setup
    k_folds = cfg.train.k_folds
    gkf = GroupKFold(n_splits=k_folds)
    
    # Containers for Global Metrics
    fold_results = []
    global_true_labels = []
    global_pred_labels = []
    
    # 4. Training Loop (Cross-Validation)
    logger.info(f"Starting Group K-Fold (k={k_folds}) Training...")
    
    # Define a readable run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.model.name}_{timestamp}"

    # Only one MLFlow run for the entire experiment (or one per fold? Usually one per experiment with nested runs or just logging mean)
    # Let's use one parent run for the experiment, and maybe log fold metrics as fold_x_metric
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Prepare inputs for splitter (X and y can be dummy, groups is what matters)
        X_dummy = np.zeros(len(dataset))
        y_dummy = np.array(dataset.labels)
        groups = np.array(dataset.actor_ids)
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=groups)):
            logger.info(f"=== Fold {fold+1}/{k_folds} ===")
            
            # Create Subsets & Loaders
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=cfg.train.batch_size, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=cfg.train.batch_size, shuffle=False)
            
            # Initialize Model & Optimizer (Fresh per fold)
            model_class = get_model_class(cfg.model.name)
            model = model_class(cfg).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
            
            best_val_acc = 0.0
            patience_counter = 0 # Early Stopping Counter
            
            # History Container for Plotting
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            # Epoch Loop
            for epoch in range(cfg.train.epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
                
                # Append History
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
                
                # Checkpointing & Early Stopping Logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0 # Reset counter
                    # Save best model for this fold
                    # Note: Hydra changes cwd to outputs/YYYY-MM-DD/HH-MM-SS, so "weights" folder is local to this run
                    save_path = f"weights/best_model_fold{fold+1}.pt" 
                    os.makedirs("weights", exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"    -> New Best Model Saved! (Acc: {val_acc:.4f})")
                else:
                    patience_counter += 1
                    logger.info(f"    -> No improvement. Patience: {patience_counter}/{cfg.train.early_stopping}")
                    
                if patience_counter >= cfg.train.early_stopping:
                    logger.info(f"Early Stopping triggered at Epoch {epoch+1}")
                    break
            
            # Plot Learning Curve for this Fold
            lc_path = f"fold_{fold+1}_learning_curve.png"
            plot_learning_curves(history, lc_path)
            mlflow.log_artifact(lc_path)
            logger.info(f"Learning Curve saved to {lc_path}")
                    
            # End of Fold: Load the BEST model (not the last one) to evaluate and log metrics
            # This ensures we report the peak performance, avoiding overfitting from later epochs
            model.load_state_dict(torch.load(f"weights/best_model_fold{fold+1}.pt"))
            _, final_acc, final_f1, final_labels, final_preds = validate(model, val_loader, criterion, device)
            
            fold_results.append({
                "fold": fold + 1,
                "accuracy": final_acc,
                "f1_score": final_f1
            })
            
            global_true_labels.extend(final_labels)
            global_pred_labels.extend(final_preds)
            
            # Log Fold Metrics to MLFlow
            mlflow.log_metric(f"fold_{fold+1}_acc", final_acc)
            mlflow.log_metric(f"fold_{fold+1}_f1", final_f1)
            
            logger.info(f"Fold {fold+1} Finished. Best Acc: {final_acc:.4f}")

    # 5. Global Evaluation
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])
    
    logger.info(f"\nTraining Complete.")
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    logger.info(f"Average Macro F1: {avg_f1:.4f}")
    
    # Log Average Metrics
    with mlflow.start_run(run_id=parent_run.info.run_id):
        mlflow.log_metric("avg_accuracy", avg_acc)
        mlflow.log_metric("avg_f1_score", avg_f1)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(global_true_labels, global_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=dataset.labels[:8], # Assumption: using indices 0-7, mapping logic needed if strings
                    yticklabels=dataset.labels[:8]) # This might be wrong if labels are just collected values. 
                    # Dataset labels are 0-7 integers. We need the string mapping.
        
 

        # Helper functions moved to module level

        # Use INV_EMOTION_MAP logic locally or imported
        emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # 1. Global Confusion Matrix Plot
        cm = confusion_matrix(global_true_labels, global_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_names, 
                    yticklabels=emotion_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Global Confusion Matrix (Acc: {avg_acc:.4f})')
        
        cm_path = "global_confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        logger.info(f"Confusion Matrix saved to {cm_path}")

        # 2. Class-wise Accuracy Plot
        class_acc_path = "global_class_accuracy.png"
        plot_class_accuracy(cm, emotion_names, class_acc_path)
        mlflow.log_artifact(class_acc_path)
        logger.info(f"Class-wise Accuracy saved to {class_acc_path}")

        # 3. t-SNE Visualization 
 
        
        # 3. t-SNE Visualization 
        # plot_tsne is now valid globally

        # Extract features for t-SNE using the Best Demo Model
        # We need to re-run validation on the full dataset (or a subset) with hooks
        logger.info("Generating t-SNE plot...")
        
        # 1. Identify and Save Best Demo Model 
        best_fold_idx = np.argmax([r['accuracy'] for r in fold_results])
        best_fold_num = fold_results[best_fold_idx]['fold']
        
        best_model_path = f"weights/best_model_fold{best_fold_num}.pt"
        demo_model_path = "weights/best_model_demo.pt"
        
        # Copy to current run dir
        # Copy to current run dir
        shutil.copy(best_model_path, demo_model_path)
        logger.info(f"Best Fold was Fold {best_fold_num}. Saved as {demo_model_path}")
        
        # Copy to Project Root "saved_models" for easy Inference access
        # Copy to Project Root "saved_models" for easy Inference access
        original_cwd = hydra.utils.get_original_cwd()
        global_save_dir = os.path.join(original_cwd, "saved_models")
        os.makedirs(global_save_dir, exist_ok=True)
        global_demo_path = os.path.join(global_save_dir, f"best_model_{cfg.model.name}.pt")
        
        shutil.copy(demo_model_path, global_demo_path)
        logger.info(f"[Global Export] Best model copied to: {global_demo_path}")

        # 2. Reload demo model for t-SNE
        # Re-initialize model class (fresh start)
        model_class = get_model_class(cfg.model.name)
        model = model_class(cfg).to(device)
        model.load_state_dict(torch.load(demo_model_path))
        model.eval()
        
        # Hook to capture features before classifier
        features_list = []
        def get_features_hook(module, input, output):
            # input is a tuple (x_in, ), x_in shape: (Batch, Hidden_Dim)
            features_list.append(input[0].detach().cpu().numpy())

        # Register hook on the classifier (it receives the global features)
        handle = model.classifier.register_forward_hook(get_features_hook)
        
        # Run inference on all data (using Full Dataset)
        full_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)
        all_labels_tsne = []
        
        with torch.no_grad():
            for inputs, labels in full_loader:
                inputs = inputs.to(device)
                model(inputs) # Hook will capture features
                all_labels_tsne.extend(labels.numpy())
                
        handle.remove()
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=0)
        
        # Plot t-SNE
        tsne_path = "global_tsne_plot.png"
        plot_tsne(all_features, all_labels_tsne, emotion_names, tsne_path)
        mlflow.log_artifact(tsne_path)
        logger.info(f"t-SNE Plot saved to {tsne_path}")
        
        logger.info(f"Best Fold was Fold {best_fold_num}. Saved as {demo_model_path} for inference.")

if __name__ == "__main__":
    main()
