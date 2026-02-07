import torch
import torch.nn as nn
import math
from src.utils.registry import register_model

# EfficientNet-Lite leverages Mobile Inverted Bottleneck Conv (MBConv) 
# BUT removes Squeeze-and-Excitation (SE) and uses ReLU6 instead of Swish.

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se=False, act_layer=nn.ReLU6):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        self.expand = in_channels != hidden_dim

        layers = []
        # Expansion phase
        if self.expand:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(act_layer(inplace=True))

        # Depthwise convolution phase
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(act_layer(inplace=True))

        # SE Block would go here, but EfficientNet-Lite removes it.
        if use_se:
            # Placeholder: In Lite, this is skipped.
            pass

        # Projection phase
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@register_model("efficientnet_lite0")
class EfficientNetLite0(nn.Module):
    def __init__(self, cfg):
        super(EfficientNetLite0, self).__init__()
        
        # EfficientNet-Lite0 definitions
        # Width multim: 1.0, Depth multim: 1.0, Res: 224 (Variable for us)
        # Stage settings similar to EfficientNet-B0 but no SE, ReLU6.
        
        self.width_coeff = 1.0
        self.depth_coeff = 1.0
        
        # Resolution is dynamic in our case (Time axis variable), but Freq is 128 (fixed)
        
        def round_filters(filters):
            multiplier = self.width_coeff
            divisor = 8
            filters *= multiplier
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)

        # Config: [kernel, stride, in_ch, out_ch, expand_ratio, repeats]
        # Based on EfficientNet-B0, adapted for Lite (No SE, ReLU6)
        block_args = [
            [3, 1, 32, 16, 1, 1],
            [3, 2, 16, 24, 6, 2],
            [5, 2, 24, 40, 6, 2],
            [3, 2, 40, 80, 6, 3],
            [5, 1, 80, 112, 6, 3],
            [5, 2, 112, 192, 6, 4],
            [3, 1, 192, 320, 6, 1]
        ]

        # Stem
        out_channels = round_filters(32)
        self.features = [
            nn.Sequential(
                nn.Conv2d(1, out_channels, 3, 2, 1, bias=False), # Stride 2, In=1 (Log-Mel)
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        ]
        
        in_channels = out_channels
        
        # Blocks
        for k, s, in_c, out_c, expand, repeats in block_args:
            out_c = round_filters(out_c)
            # Stride only applies to the first block of the sequence
            for i in range(repeats):
                stride = s if i == 0 else 1
                self.features.append(
                    MBConvBlock(in_channels, out_c, k, stride, expand, use_se=False, act_layer=nn.ReLU6)
                )
                in_channels = out_c
                
        # Head
        last_channels = round_filters(1280)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_channels, last_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU6(inplace=True)
            )
        )
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(last_channels, 8) # 8 Emotions
        
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
