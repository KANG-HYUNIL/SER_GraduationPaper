import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from src.utils.registry import register_model

@register_model("cnn_temporal_attention")
class TemporalAttentionCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # 1. Feature Extractor (Same as Baseline but NO Global Pool)
        n_mels = cfg.data.n_mels
        num_classes = 8
        hidden_dims = cfg.model.get("hidden_dims", [64, 128, 256, 512])
        attention_dim = cfg.model.get("attention_dim", 128)
        dropout_prob = cfg.model.get("dropout", 0.1)
        
        layers = []
        in_channels = 1
        
        for out_channels in hidden_dims:
            layers.append(self._build_block(in_channels, out_channels))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # 2. Attention Module
        # Input features: (Batch, C, Freq, Time)
        # We pool Frequency dimension first -> (Batch, C, Time)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        # Attention Mechanism: Calculate score for each time step
        # Query: Maps feature channel (C) to attention score (1)
        self.attention_layer = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, 1, kernel_size=1)
        )
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
    def _build_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x: (Batch, 1, 128, 512)
        
        # 1. Extract Features
        x = self.features(x)
        # x: (Batch, 512, 8, 32)
        
        # 2. Prepare for Attention
        # Frequency(H') 차원만 압축하여 시간 축 정보는 보존함.
        x_time = self.freq_pool(x).squeeze(2) 
        # x_time: (Batch, 512, 32)
        
        # 3. Calculate Attention Scores
        # 각 시간 스텝(32개)에 대한 가중치 계산
        scores = self.attention_layer(x_time)
        alpha = F.softmax(scores, dim=2) 
        
        # 4. Weighted Sum (Context Vector)
        # 중요한 시간대의 특징에 더 높은 가중치를 부여하여 합산
        context = torch.sum(x_time * alpha, dim=2)
        
        # 5. Classification
        out = self.classifier(context)
        return out
