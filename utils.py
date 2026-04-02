"""
Utility functions for Chinese License Plate Recognition.

This module provides essential utilities including:
- Character-to-index mapping
- CTC decoding (single and batch)
- GPU memory monitoring
- Ultralytics-style experiment management
- Results visualization (CSV, plots, YAML)
"""

import os
import csv
from typing import Dict, List, Tuple

import torch
import numpy as np


def char2idx(plate_chars: List[str]) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Create character to index mapping dictionaries.
    
    Args:
        plate_chars: List of license plate characters
        
    Returns:
        char2idx: Character to index dictionary ('<blank>' -> 0)
        idx2char: Index to character dictionary
        num_classes: Total number of character classes (including blank)
    
    Example:
        >>> chars = ['京', 'A', '1']
        >>> c2i, i2c, n = char2idx(chars)
        >>> print(c2i)  # {'<blank>': 0, '京': 1, 'A': 2, '1': 3}
    """
    char2idx = {ch: i+1 for i, ch in enumerate(plate_chars)}
    char2idx['<blank>'] = 0
    idx2char = {v: k for k, v in char2idx.items()}
    num_classes = len(char2idx)
    return char2idx, idx2char, num_classes


def ctc_decode(logits, idx2char, blank_idx=0):
    """
    logits: (T, num_classes) 模型输出(未softmax)
    idx2char: 索引->字符字典
    返回: (text, confidence)
    """
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    chars, confs = [], []
    prev = None
    for t, p in enumerate(preds):
        if p != blank_idx and p != prev:
            chars.append(idx2char[p])
            confs.append(probs[t, p].item())
        prev = p
    text = ''.join(chars)
    avg_conf = np.mean(confs) if confs else 0.0
    return text, avg_conf


def ctc_decode_batch(logits, idx2char, blank_idx=0):
    texts, confs = [], []
    for i in range(logits.size(0)):
        t, c = ctc_decode(logits[i], idx2char, blank_idx)
        texts.append(t)
        confs.append(c)
    return texts, confs


def get_gpu_memory():
    """获取GPU显存使用量 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def format_time(seconds):
    """格式化时间为 HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def increment_path(base_dir, exp_name):
    """
    参考 Ultralytics 的实验目录自动递增逻辑。
    首次使用 <base_dir>/<exp_name>，已存在则递增为 <exp_name>2、<exp_name>3 ...
    """
    path = os.path.join(base_dir, exp_name)
    if not os.path.exists(path):
        return path
    n = 2
    while os.path.exists(os.path.join(base_dir, f'{exp_name}{n}')):
        n += 1
    return os.path.join(base_dir, f'{exp_name}{n}')


# CSV 列头顺序，与 Ultralytics results.csv 风格保持一致
CSV_FIELDNAMES = [
    'epoch', 'train/ctc_loss',
    'val/ctc_loss', 'val/seq_acc', 'val/char_acc',
    'lr'
]


def save_results_csv(csv_path, row: dict, write_header=False):
    """
    追加一行指标到 results.csv。
    row 须包含 CSV_FIELDNAMES 中的所有键。
    """
    mode = 'w' if write_header else 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in CSV_FIELDNAMES})


def plot_results(csv_path, save_path):
    """
    读取 results.csv，绘制 Loss / Accuracy / LR 曲线并保存为 results.png。
    仅在 matplotlib 可用时执行，否则静默跳过。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # 无界面后端
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training Results', fontsize=14)

    # Loss
    axes[0].plot(df['epoch'], df['train/ctc_loss'], label='train', color='#1f77b4')
    axes[0].plot(df['epoch'], df['val/ctc_loss'],   label='val',   color='#ff7f0e')
    axes[0].set_title('CTC Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(df['epoch'], df['val/seq_acc'],  label='seq_acc',  color='#2ca02c')
    axes[1].plot(df['epoch'], df['val/char_acc'], label='char_acc', color='#d62728')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(df['epoch'], df['lr'], color='#9467bd')
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_args_yaml(yaml_path, train_cfg, model_cfg):
    """
    将训练配置和模型配置序列化为 args.yaml，类似 Ultralytics 的 args.yaml。
    """
    try:
        import yaml
    except ImportError:
        return

    data = {
        'train': {
            'data_dir':      train_cfg.data_dir,
            'train_label':   train_cfg.train_label,
            'val_label':     train_cfg.val_label,
            'batch_size':    train_cfg.batch_size,
            'epochs':        train_cfg.epochs,
            'lr':            train_cfg.lr,
            'weight_decay':  train_cfg.weight_decay,
            'warmup_epochs': train_cfg.warmup_epochs,
            'num_workers':   train_cfg.num_workers,
            'device':        train_cfg.device,
            'save_dir':      train_cfg.save_dir,
            'exp_name':      train_cfg.exp_name,
            'save_interval': train_cfg.save_interval,
        },
        'model': {
            'small':      model_cfg.small,
            'rnn_hidden': model_cfg.rnn_hidden,
            'rnn_layers': model_cfg.rnn_layers,
            'img_height': model_cfg.img_height,
            'max_width':  model_cfg.max_width,
            'num_chars':  len(model_cfg.plate_chars),
        }
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
