import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.utils.registry import register_model

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Position Attention)
    Inspired by CBAM (Convolutional Block Attention Module).
    
    특징 맵 내에서 어떤 '위치'가 감정 정보가 많은지를 학습하는 레이어입니다.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 입력된 커널 크기에 따라 패딩을 설정하여 해상도를 유지함
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # MaxPool과 AvgPool 결과를 결합하므로 입력 채널은 2가 됨
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 채널 축(dim=1)을 따라 평면(Max/Avg) 생성
        # torch.mean(x, dim=1, keepdim=True) -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # torch.max(x, dim=1, keepdim=True)[0] -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 두 가지 특징을 결합: (B, 2, H, W)
        x_mapped = torch.cat([avg_out, max_out], dim=1)
        
        # 3. Convolution을 통해 위치별 중요도(가중치) 산출
        x_mapped = self.conv(x_mapped)
        
        # 4. Sigmoid를 통해 0~1 사이 가중치 부여 후 원본에 곱함 (Attention Scale)
        return x * self.sigmoid(x_mapped)

@register_model("cnn_spatial_attention")
class SpatialAttentionCNN(nn.Module):
    """
    공간 어텐션(Spatial Attention)이 적용된 CNN 모델입니다.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        n_mels = cfg.data.n_mels
        num_classes = 8
        hidden_dims = cfg.model.get("hidden_dims", [64, 128, 256, 512])
        dropout_prob = cfg.model.get("dropout", 0.1)
        
        layers = []
        in_channels = 1
        
        # Conv Blocks와 Spatial Attention 결합
        for out_channels in hidden_dims:
            layers.append(self._build_spatial_block(in_channels, out_channels))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # 최종 분류기: 입력 사이즈가 고정됨(Bicubic Resizing)에 따라 GAP를 제거하고 
        # Flatten 후 Linear 로직으로 변경 가능하나, 범용성을 위해 우선 Flatten만 적용
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 채널별 핵심 특징만 남김 (전체 공간 뭉개기 지양 목적)
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
    def _build_spatial_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SpatialAttention() # 위치(공간) 어텐션 추가
        )

    def forward(self, x):
        # x shape: (B, 1, 128, 512)
        x = self.features(x)
        # x shape: (B, 512, 8, 32)
        out = self.classifier(x)
        return out
