"""
CRNN (Convolutional Recurrent Neural Network) for License Plate Recognition.

This module implements a CRNN architecture combining CNN feature extraction 
with bidirectional LSTM sequence modeling for sequence recognition tasks.

Reference:
    Shi, B., Bai, X., & Yao, C. (2016). An End-to-End Trainable Neural Network 
    for Image-based Sequence Recognition and Its Application to Scene Text 
    Recognition. IEEE TPAMI.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN 网络架构，用于序列识别任务。
    
    结构组成:
        - CNN: 多层卷积 + 池化提取视觉特征
        - BiLSTM: 双向 LSTM 进行序列建模
        - FC: 全连接层输出字符概率分布
    
    Args:
        num_classes: 字符类别数量（包含 blank 类别）
        rnn_hidden: LSTM 隐藏层维度
        rnn_layers: LSTM 层数
        small: 是否使用轻量级模型（通道数减半）
    
    Input:
        x: 灰度图像张量 (B, 1, H, W)
    
    Output:
        logits: 每帧的字符概率分布 (B, T, num_classes)
    """
    
    def __init__(self, num_classes: int, rnn_hidden: int = 256, 
                 rnn_layers: int = 2, small: bool = False):
        super().__init__()
        
        # 根据模型规模配置通道数
        if small:
            channels = [32, 64, 128]
            rnn_hidden = 128
        else:
            channels = [64, 128, 256]
        
        # ── CNN 特征提取器 ────────────────────────────────────────
        cnn_layers = []
        in_ch = 1
        
        # 三层标准卷积块
        for out_ch in channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 高度减半
            ])
            in_ch = out_ch
        
        # 额外卷积层（保持通道数不变）
        cnn_layers.extend([
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), 
                        padding=(0, 1))  # 宽度方向步长 1，高度方向步长 2
        ])
        
        # 自适应池化：强制高度为 1，宽度保持不变
        # cnn_layers.append(nn.AdaptiveAvgPool2d((1, None)))
        cnn_layers.extend([
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        ])
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # ── RNN 序列建模 ────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if rnn_layers > 1 else 0.0
        )
        
        # ── 分类层 ──────────────────────────────────────────────
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # CNN 特征提取: (B, 1, H, W) → (B, C, 1, W')
        conv = self.cnn(x)
        
        # 重塑为序列格式：(B, W', C)
        seq = conv.squeeze(2).permute(0, 2, 1)
        
        # LSTM 序列建模：(B, W', C) → (B, W', hidden*2)
        out, _ = self.lstm(seq)
        
        # 全连接分类：(B, W', hidden*2) → (B, W', num_classes)
        out = self.fc(out)
        
        return out


if __name__ == '__main__':
    # 模型测试
    model = CRNN(num_classes=78, small=True)
    print(model)
    
    # 前向传播测试
    x = torch.randn(1, 1, 32, 128)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
