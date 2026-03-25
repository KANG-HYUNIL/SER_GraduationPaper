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
        
        # Spatial Attention 모듈 추가 (기존 베이스라인에 선택적 추가 가능하도록 구성)
        self.spatial_attn = SpatialAttention() if cfg.model.get("use_spatial", False) else nn.Identity()

        # Global Average Pooling (1x1)은 시공간 정보를 모두 뭉개버리므로 가급적 지양.
        # 고정 해상도(8x32)를 최대한 활용하기 위해 pooling 사이즈를 키우거나 Flatten을 사용함.
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # 8x32 -> 4x4 정도로 압축하여 정보 손실 최소화
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[-1] * 4 * 4, num_classes) # Pooled 사이즈 반영 (512 * 4 * 4)
        )
        
    def _build_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), #kernel size
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # kernel size 
        )

    def forward(self, x):
        # x shape: (Batch, 1, 128, 512)
        x = self.features(x)
        # x shape: (Batch, 512, 8, 32)
        
        x = self.spatial_attn(x) # Spatial Attention 적용
        
        x = self.pool(x)
        # x shape: (Batch, 512, 4, 4)
        
        out = self.classifier(x)
        return out
