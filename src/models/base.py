import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.utils.registry import register_model

@register_model("cnn_baseline")
class BaselineCNN(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # Unpack config
        n_mels = cfg.data.n_mels
        num_classes = 8 # RAVDESS has 8 emotions
        hidden_dims = cfg.model.get("hidden_dims", [64, 128, 256, 512])
        dropout_prob = cfg.model.get("dropout", 0.1)
        
        layers = []
        in_channels = 1 # Log-Mel Spectrogram (1 Channel)
        
        # Build 4 Conv Blocks
        for out_channels in hidden_dims:
            layers.append(self._build_block(in_channels, out_channels))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # Global Average Pooling to handle variable time lengths
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
    def _build_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), #kernel size
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # kernel size 
        )

    def forward(self, x):
        # x shape: (Batch, 1, n_mels, Time)
        x = self.features(x)
        # x shape: (Batch, Last_Dim, H', W')
        
        x = self.global_pool(x)
        # x shape: (Batch, Last_Dim, 1, 1)
        
        out = self.classifier(x)
        # out shape: (Batch, Num_Classes)
        return out
