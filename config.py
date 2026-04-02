"""
Chinese License Plate Recognition - Configuration
中文车牌识别系统配置文件
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # CRNN 模型参数
    rnn_hidden: int = 256
    rnn_layers: int = 2
    small: bool = True  # 使用轻量级模型
    
    # 输入图像尺寸
    img_height: int = 32
    max_width: int = 128
    
    # 字符集（索引0保留给CTC blank）
    plate_chars: List[str] = field(default_factory=lambda: [
        '京','津','沪','渝','冀','晋','辽','吉','黑','苏',
        '浙','皖','闽','赣','鲁','豫','鄂','湘','粤','琼',
        '川','贵','云','陕','甘','青','蒙','桂','宁','新','藏',
        'A','B','C','D','E','F','G','H','J','K','L','M','N','P',
        'Q','R','S','T','U','V','W','X','Y','Z',
        '0','1','2','3','4','5','6','7','8','9',
        '警','学','挂','港','澳','使','领','险'
    ])


@dataclass
class TrainConfig:
    """训练配置"""
    # 数据路径
    data_dir: str = './license_plate_data'
    train_label: str = './license_plate_data/train.txt'
    val_label: str = './license_plate_data/val.txt'
    
    # 训练参数
    batch_size: int = 256
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # 数据加载
    num_workers: int = 4
    drop_last: bool = True
    
    # 保存和日志
    # 每次训练结果保存在 save_dir/<exp_name>/ 下，内含 weights/、results.csv、args.yaml、results.png
    save_dir: str = './runs'
    exp_name: str = 'train'      # 实验名，自动递增为 train / train2 / train3 ...
    log_interval: int = 10       # 每隔多少 batch 在进度条打印一次行日志
    save_interval: int = 5       # 每隔多少 epoch 保存一次 last.pt
    
    # 设备
    device: str = 'cuda'  # 'cuda' 或 'cpu'


@dataclass
class InferConfig:
    """推理配置"""
    # 模型路径（与新目录结构 runs/train/weights/best.pt 对应）
    model_path: str = './runs/train/weights/best.pt'

    # 设备
    device: str = 'cuda'

# 颜色定义 for YOLO-style logging
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# 默认配置实例
model_cfg = ModelConfig()
train_cfg = TrainConfig()
infer_cfg = InferConfig()
