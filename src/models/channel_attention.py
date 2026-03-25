import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.utils.registry import register_model

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention)
    References:
        - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
        - "Channel Attention Scale Feature Fusion..." for SER.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Pooling -> (B, C, 1, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn weights -> (B, C) -> (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Rescale features
        return x * y.expand_as(x)

@register_model("cnn_channel_attention")
class ChannelAttentionCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        n_mels = cfg.data.n_mels
        num_classes = 8
        hidden_dims = cfg.model.get("hidden_dims", [64, 128, 256, 512])
        reduction_ratio = cfg.model.get("reduction_ratio", 16)
        dropout_prob = cfg.model.get("dropout", 0.1)
        
        layers = []
        in_channels = 1
        
        # Build Conv Blocks with SE-Block integrated
        for out_channels in hidden_dims:
            layers.append(self._build_se_block(in_channels, out_channels, reduction_ratio))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # Classifier (Final Stage)
        # 데이터를 심하게 뭉개는 1x1 Pooling 대신 4x4 정도의 해상도를 유지함
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[-1] * 4 * 4, num_classes)
        )
        
    def _build_se_block(self, in_c, out_c, reduction):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SEBlock(out_c, reduction) # 채널 어텐션은 채널 정보를 모아야 하므로 내부 pooling은 필수
        )

    def forward(self, x):
        # x: (B, 1, 128, 512)
        x = self.features(x)
        # x: (B, 512, 8, 32)
        x = self.pool(x)
        # x: (B, 512, 4, 4)
        out = self.classifier(x)
        return out
