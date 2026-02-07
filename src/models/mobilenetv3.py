import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from src.utils.registry import register_model

@register_model("mobilenet_v3_small")
class MobileNetV3Small(nn.Module):
    def __init__(self, cfg):
        super(MobileNetV3Small, self).__init__()
        
        # Load pre-trained MobileNetV3 Small
        # We use weights=None to strictly follow "paper architecture" without transfer learning bias implies
        # but user might want ImageNet weights. 
        # Given "standard paper implementation" context for SER, usually fine-tuning is better.
        # But let's start with weights=None (random init) or ImageNet weights if specified.
        # For now, let's stick to standard architecture.
        # However, torchvision's implementation is standard.
        
        # Using default weights=None for now as per "propose structure" request, not necessarily transfer learning.
        self.model = mobilenet_v3_small(weights=None)
        
        # 1. Modify First Layer to accept 1 channel (Log-Mel Spectrogram)
        # Original: nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_layer = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        # 2. Modify Classifier to output 8 classes (Emotions)
        # MobileNetV3 Small classifier structure:
        # Sequential(
        #   (0): Linear(in_features=576, out_features=1024, bias=True)
        #   (1): Hardswish()
        #   (2): Dropout(p=0.2, inplace=True)
        #   (3): Linear(in_features=1024, out_features=1000, bias=True)
        # )
        num_features = self.model.classifier[0].in_features # 576
        # We need to replace the entire classifier or just the last layer? 
        # The paper uses a specific head. Torchvision implements it. 
        # We just need to change the final output to 8.
        
        # However, to be safe and standard for 8 classes:
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 8)

    def forward(self, x):
        # x shape: (B, 1, F, T) -> (B, 1, 128, T)
        return self.model(x)
