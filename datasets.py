"""
Dataset module for Chinese License Plate Recognition.

This module provides PyTorch Dataset implementation for license plate images
with data augmentation and automatic padding/collation.
"""

import os
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import model_cfg


class PlateDataset(Dataset):
    """
    中文车牌数据集类，支持数据增强和自动 collate。
    
    数据格式要求:
        - label_file 每行格式：image_path label
        - 图像命名：车牌号_序号.jpg（如：京 A12345_0.jpg）
        - 支持灰度/彩色图像（自动转换为灰度）
    
    Args:
        label_path: 标注文件路径
        plate_chars: 字符集列表（None 则使用配置默认值）
        img_height: 目标图像高度
        max_width: 最大图像宽度
        is_train: 是否为训练模式（启用数据增强）
        use_aug: 是否使用数据增强
    
    Returns:
        tuple: (image_tensor, label_indices, length)
    """
    
    # 字符集（索引 0 保留给 CTC blank）
    PLATE_CHARS = model_cfg.plate_chars
    BLANK_IDX = 0
    
    def __init__(self, label_path: str, 
                 plate_chars: Optional[List[str]] = None,
                 img_height: Optional[int] = None,
                 max_width: Optional[int] = None,
                 is_train: bool = True, 
                 use_aug: bool = True):
        
        # 使用配置文件中的默认值
        self.img_height = img_height if img_height is not None else model_cfg.img_height
        self.max_width = max_width if max_width is not None else model_cfg.max_width
        self.is_train = is_train
        self.use_aug = use_aug and is_train
        
        if plate_chars is not None:
            self.PLATE_CHARS = plate_chars
        
        # 构建字符映射表
        self.char2idx = {ch: i + 1 for i, ch in enumerate(self.PLATE_CHARS)}
        self.char2idx['<blank>'] = self.BLANK_IDX
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.num_classes = len(self.char2idx)
        
        # 加载数据列表
        self.images: List[str] = []
        self.labels: List[str] = []
        self._load_annotations(label_path)
        
        print(f"Loaded {len(self.images)} samples. "
              f"Char set size: {self.num_classes}")
    
    def _load_annotations(self, label_path: str):
        """加载标注文件"""
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.images.append(parts[0])
                    self.labels.append(parts[1])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """获取单个样本"""
        img = self._load_image(idx)
        
        if self.use_aug:
            img = self._augment(img)
        
        img = self._resize_normalize(img)
        img = (img.astype(np.float32) / 255.0)[np.newaxis, ...]  # (1, H, W)
        img_tensor = torch.from_numpy(img).float()
        
        # 标签编码
        label = self.labels[idx]
        label_indices = [self.char2idx[ch] for ch in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return img_tensor, label_tensor, len(label_indices)
    
    def _load_image(self, idx: int) -> np.ndarray:
        """加载图像并转换为灰度"""
        path = self.images[idx]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img
    
    def _resize_normalize(self, img: np.ndarray) -> np.ndarray:
        """保持宽高比 resize 到固定高度"""
        h, w = img.shape
        
        # 固定高度，保持宽高比
        ratio = self.img_height / h
        new_w = int(w * ratio)
        
        # 限制最大宽度
        if new_w > self.max_width:
            new_w = self.max_width
        
        return cv2.resize(img, (new_w, self.img_height))
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """数据增强"""
        # 亮度对比度调整 (50% 概率)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 高斯噪声 (30% 概率)
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # 高斯模糊 (20% 概率)
        if random.random() > 0.8:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        return img
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        功能:
            - 将 batch 中的图像 padding 到相同宽度
            - 拼接所有标签序列
            - 记录每个标签的长度
        
        Args:
            batch: [(image, label, length), ...]
        
        Returns:
            images: (N, 1, H, W_max) padded images
            labels: (N*L,) concatenated labels
            lengths: (N,) sequence lengths
        """
        images, labels, lengths = zip(*batch)
        
        # 找到最大宽度
        max_w = max(img.shape[2] for img in images)
        
        # Padding 图像
        padded = []
        for img in images:
            _, h, w = img.shape
            if w < max_w:
                pad_img = torch.zeros(1, h, max_w)
                left = (max_w - w) // 2
                pad_img[:, :, left:left + w] = img
                img = pad_img
            padded.append(img)
        
        images = torch.stack(padded, dim=0)
        labels = torch.cat(labels, dim=0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        return images, labels, lengths


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # 测试代码
    train_dataset = PlateDataset(
        label_path=r'./license_plate_data/train.txt',
        img_height=32,
        max_width=128,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=PlateDataset.collate_fn,
        num_workers=1
    )
    
    # 打印第一个批次信息
    for images, labels, lengths in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Lengths: {lengths}")
        break