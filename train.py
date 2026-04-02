import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from datasets import PlateDataset
from model import CRNN
from utils import (ctc_decode_batch, get_gpu_memory, format_time,
                   increment_path, save_results_csv, plot_results, save_args_yaml)
from config import train_cfg, model_cfg, Colors

# ── 从配置文件加载参数 ──────────────────────────────────────────────────
LABEL_PATH    = train_cfg.train_label
VAL_LABEL_PATH = train_cfg.val_label
BATCH_SIZE    = train_cfg.batch_size
EPOCHS        = train_cfg.epochs
LR            = train_cfg.lr
WEIGHT_DECAY  = train_cfg.weight_decay
WARMUP_EPOCHS = train_cfg.warmup_epochs
DEVICE        = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
LOG_INTERVAL  = train_cfg.log_interval
SAVE_INTERVAL = train_cfg.save_interval
NUM_WORKERS   = train_cfg.num_workers
DROP_LAST     = train_cfg.drop_last

IMG_HEIGHT  = model_cfg.img_height
MAX_WIDTH   = model_cfg.max_width
MODEL_SMALL = model_cfg.small


# ── 学习率调度 ──────────────────────────────────────────────────────────
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, min_factor=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_factor + (1 - min_factor) * (1 + np.cos(np.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── 单轮训练 ────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    epoch_start = time.time()

    # YOLO-style 表头（每10轮刷新一次）
    if epoch == 1 or epoch % 10 == 0:
        header = (f"{'Epoch':>10}{'GPU_mem':>10}{'ctc_loss':>12}"
                  f"{'instances':>12}{'img_size':>10}")
        print(f"\n{Colors.CYAN}{header}{Colors.ENDC}")

    pbar = tqdm(loader, desc=f'  {epoch}/{EPOCHS}',
                bar_format='  {l_bar}{bar:15}{r_bar}', leave=False)

    for batch_idx, (imgs, labels, lengths) in enumerate(pbar):
        imgs    = imgs.to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)

        batch_size       = imgs.size(0)
        img_h, img_w     = imgs.shape[2], imgs.shape[3]

        logits = model(imgs)                                   # (B, T, C)
        logits = logits.permute(1, 0, 2).log_softmax(2)       # (T, B, C)
        t_len  = torch.full((logits.size(1),), logits.size(0),
                            dtype=torch.long, device=device)
        loss = criterion(logits, labels, t_len, lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()
        gpu_mem = get_gpu_memory()
        mem_str = f'{gpu_mem:.2f}G' if torch.cuda.is_available() else '0G'

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mem': mem_str})

        # 行日志（每 LOG_INTERVAL 个 batch 打印一行，配合进度条 leave=False）
        if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == num_batches:
            print(f"  {epoch:>4}/{EPOCHS}"
                  f"  {mem_str:>8}"
                  f"  {loss.item():>10.4f}"
                  f"  {batch_size:>10}"
                  f"  {img_h}x{img_w}")

    return total_loss / num_batches, time.time() - epoch_start


# ── 验证 ────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device, dataset):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_gts  = [], []

    with torch.no_grad():
        for imgs, labels, lengths in tqdm(loader, desc='  Validation',
                                          bar_format='  {l_bar}{bar:15}{r_bar}',
                                          leave=False):
            imgs    = imgs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits = model(imgs)
            logits = logits.permute(1, 0, 2).log_softmax(2)
            t_len  = torch.full((logits.size(1),), logits.size(0),
                                dtype=torch.long, device=device)
            loss = criterion(logits, labels, t_len, lengths)
            total_loss += loss.item()

            pred_texts, _ = ctc_decode_batch(logits.permute(1, 0, 2), dataset.idx2char)

            idx = 0
            for i in range(imgs.size(0)):
                length  = lengths[i].item()
                gt_text = ''.join([dataset.idx2char[c]
                                   for c in labels[idx:idx+length].tolist()])
                all_preds.append(pred_texts[i])
                all_gts.append(gt_text)
                if pred_texts[i] == gt_text:
                    correct += 1
                idx += length

    seq_acc = correct / len(loader.dataset)

    char_correct = char_total = 0
    for pred, gt in zip(all_preds, all_gts):
        for p, g in zip(pred, gt):
            if p == g:
                char_correct += 1
        char_total += len(gt)
    char_acc = char_correct / char_total if char_total > 0 else 0.0

    return total_loss / len(loader), seq_acc, char_acc


# ── 主流程 ──────────────────────────────────────────────────────────────
def main():
    # ── 创建实验目录（自动递增，如 train / train2 / train3）──────────
    exp_dir     = increment_path(train_cfg.save_dir, train_cfg.exp_name)
    weights_dir = os.path.join(exp_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    csv_path  = os.path.join(exp_dir, 'results.csv')
    yaml_path = os.path.join(exp_dir, 'args.yaml')
    png_path  = os.path.join(exp_dir, 'results.png')

    # 保存本次训练超参数
    save_args_yaml(yaml_path, train_cfg, model_cfg)

    # ── 打印训练 Banner ──────────────────────────────────────────────
    print(f"\n{Colors.GREEN}{Colors.BOLD}"
          f"Chinese License Plate Recognition — CRNN Training"
          f"{Colors.ENDC}")
    print(f"{Colors.BLUE}  Device    : {DEVICE}{Colors.ENDC}")
    print(f"{Colors.BLUE}  Epochs    : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}{Colors.ENDC}")
    print(f"{Colors.BLUE}  Results   : {exp_dir}{Colors.ENDC}")
    print(f"{Colors.BLUE}  Weights   : {weights_dir}{Colors.ENDC}\n")

    # ── 数据集 ───────────────────────────────────────────────────────
    train_dataset = PlateDataset(LABEL_PATH, is_train=True)
    val_dataset   = PlateDataset(VAL_LABEL_PATH, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=PlateDataset.collate_fn,
                              num_workers=NUM_WORKERS, drop_last=DROP_LAST)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=PlateDataset.collate_fn,
                              num_workers=NUM_WORKERS)

    print(f"{Colors.CYAN}  Train: {len(train_dataset)} samples  |  "
          f"Val: {len(val_dataset)} samples{Colors.ENDC}\n")

    # ── 模型 ─────────────────────────────────────────────────────────
    model     = CRNN(num_classes=train_dataset.num_classes, small=MODEL_SMALL).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    best_acc   = 0.0
    start_time = time.time()

    # ── 训练循环 ─────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        train_loss, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch)

        val_loss, seq_acc, char_acc = validate(
            model, val_loader, criterion, DEVICE, val_dataset)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── 每轮汇总行（参考 ultralytics 紧凑风格）────────────────
        flag = f"{Colors.GREEN}*{Colors.ENDC}" if seq_acc > best_acc else ' '
        print(f"\n{Colors.GREEN}{'─' * 72}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Epoch {epoch}/{EPOCHS}{Colors.ENDC}"
              f"  train_loss={Colors.CYAN}{train_loss:.4f}{Colors.ENDC}"
              f"  val_loss={Colors.CYAN}{val_loss:.4f}{Colors.ENDC}"
              f"  seq_acc={Colors.YELLOW}{seq_acc:.4f}{Colors.ENDC}"
              f"  char_acc={Colors.YELLOW}{char_acc:.4f}{Colors.ENDC}"
              f"  lr={current_lr:.2e}"
              f"  time={format_time(epoch_time)}  {flag}")
        print(f"{Colors.GREEN}{'─' * 72}{Colors.ENDC}\n")

        # ── 写入 results.csv ──────────────────────────────────────
        save_results_csv(csv_path,
                         row={
                             'epoch':           epoch,
                             'train/ctc_loss':  round(train_loss, 6),
                             'val/ctc_loss':    round(val_loss,   6),
                             'val/seq_acc':     round(seq_acc,    6),
                             'val/char_acc':    round(char_acc,   6),
                             'lr':              round(current_lr, 8),
                         },
                         write_header=(epoch == 1))

        # ── 实时更新曲线图 ────────────────────────────────────────
        plot_results(csv_path, png_path)

        # ── 保存权重 ──────────────────────────────────────────────
        ckpt = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'seq_acc':              seq_acc,
            'char_acc':             char_acc,
        }

        # last. — 每隔 SAVE_INTERVAL 轮覆盖
        if epoch % SAVE_INTERVAL == 0:
            torch.save(ckpt, os.path.join(weights_dir, 'last.pt'))

        # best.pt — 精度提升则覆盖
        if seq_acc > best_acc:
            best_acc = seq_acc
            torch.save(ckpt, os.path.join(weights_dir, 'best.pt'))
            print(f"  {Colors.GREEN}✓ New best  seq_acc={seq_acc:.4f}  "
                  f"char_acc={char_acc:.4f}  → saved best.pt{Colors.ENDC}\n")

    # ── 训练结束 ─────────────────────────────────────────────────────
    # 最后再保存一次 last.pt
    torch.save(ckpt, os.path.join(weights_dir, 'last.pt'))
    # 最终曲线图
    plot_results(csv_path, png_path)

    total_time = time.time() - start_time
    print(f"\n{Colors.GREEN}{Colors.BOLD}Training complete!{Colors.ENDC}")
    print(f"  Best seq_acc : {Colors.CYAN}{best_acc:.4f}{Colors.ENDC}")
    print(f"  Total time   : {Colors.CYAN}{format_time(total_time)}{Colors.ENDC}")
    print(f"  Results saved: {Colors.YELLOW}{exp_dir}{Colors.ENDC}")
    print(f"    ├── args.yaml")
    print(f"    ├── results.csv")
    print(f"    ├── results.png")
    print(f"    └── weights/")
    print(f"        ├── best.pt")
    print(f"        └── last.pt")


if __name__ == '__main__':
    main()
