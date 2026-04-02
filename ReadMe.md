# Chinese License Plate Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/eecn/chinese_license_plate_recognition?style=social)](https://github.com/eecn/chinese_license_plate_recognition/stargazers)

**基于 CRNN 的中文车牌识别系统，采用 Ultralytics 风格的工程化设计。**

本项目实现了端到端的中文车牌号码识别，支持普通蓝牌、新能源绿牌、特种车牌等多种类型。采用改进的 CRNN（Convolutional Recurrent Neural Network）架构，结合 CTC Loss 进行序列识别训练，具备高精度和实时推理能力。

🔗 **项目地址**: [https://github.com/eecn/chinese_license_plate_recognition](https://github.com/eecn/chinese_license_plate_recognition)

> ⚠️ **免责声明**: 本项目及提供的数据集仅供学习和研究使用，不得用于任何商业用途。数据来源包括公开互联网资源和作者标注数据。

## ✨ 特性亮点

- **🎯 Ultralytics 风格**: 参考 YOLO 系列的工程化设计，输出 `results.csv`、`args.yaml`、训练曲线等标准产物
- **📊 可视化监控**: 自动绘制 Loss/Accuracy/LR 曲线图，无需额外工具
- **⚡ 高性能模型**: 提供轻量级 (Small) 和标准版两种配置，平衡速度与精度
- **🔧 配置中心化**: 所有超参数集中在 `config.py`，支持实验目录自动递增
- **🎨 数据增强**: 内置亮度调整、噪声注入、高斯模糊等增强策略
- **🚀 开箱即用**: 完整的训练 - 验证 - 推理流程，包含预训练权重即可快速部署

---

## 📦 项目结构

```
chinese_license_plate_recognition/
├── checkpoints/              # 旧版模型保存目录（兼容）
├── runs/                     # 新版训练输出目录 (Ultralytics 风格)
│   ├── train/                # 第一次训练结果
│   │   ├── args.yaml         # 超参数快照
│   │   ├── results.csv       # 每轮指标记录
│   │   ├── results.png       # 可视化曲线图
│   │   └── weights/          # 模型权重
│   │       ├── best.pt       # 最佳模型
│   │       └── last.pt       # 最新检查点
├── license_plate_data/       # 数据集目录
│   ├── images/               # 车牌图像
│   ├── char_cnt.py           # 字符统计工具
│   ├── data_pre.py           # 数据预处理脚本
│   ├── train.txt             # 训练集标注
│   └── val.txt               # 验证集标注
├── config.py                 # 全局配置文件
├── datasets.py               # PyTorch Dataset 实现
├── infer.py                  # 推理脚本
├── model.py                  # CRNN 模型定义
├── requirements.txt          # Python 依赖
├── train.py                  # 训练脚本
├── utils.py                  # 工具函数
├── test.jpg                  # 测试样例
└── ReadMe.md                 # 项目文档
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/eecn/chinese_license_plate_recognition.git
cd chinese_license_plate_recognition

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

**依赖说明：**
- `torch>=1.9.0`: 深度学习框架
- `opencv-python`: 图像处理
- `tqdm`: 进度条显示
- `matplotlib`: 曲线绘制
- `pandas`: CSV 数据处理
- `pyyaml`: YAML 配置解析

### 2. 数据准备

#### 📥 数据集获取

**重要提示**: 由于数据集文件较大，无法通过 GitHub 分发。如需获取数据集，请通过以下方式联系：

- **邮箱**: [联系邮箱](mailto:你的实际邮箱)  
- **邮件主题**: 申请车牌识别数据集 - [你的姓名/单位]  
- **用途说明**: 请简要说明使用目的（如学术研究、学习交流等）

**数据集信息**:
- **图像数量**: 81,720 张
- **格式**: JPG 图像 + TXT 标注文件

> ⚠️ **免责声明**: 本数据集仅供学习和研究使用，不得用于任何商业用途。

#### 数据格式要求

- **图像命名**: `车牌号_序号.jpg`（如：`京 A12345_0.jpg`）
- **图像内容**: 单排车牌（双排车牌需预先上下拼接）
- **标注文件**: 每行格式为 `图像路径 车牌号`

示例：
```
license_plate_data/images/京A12345_0.jpg 京 A12345
license_plate_data/images/沪 B67890_1.jpg 沪 B67890
```
license_plate_data共81720张车牌数据，覆盖所有标签字符。该数据集为互联网和我自己标注的一些数据，如有需要请通过邮箱联系我获取数据。

#### 数据预处理

```bash
cd license_plate_data

# 统计字符分布（可选）
python char_cnt.py

# 划分训练集/验证集（8:2 比例）
python data_pre.py
```

生成的 `train.txt` 和 `val.txt` 将保存在 `license_plate_data/` 目录下。

### 3. 模型训练

```bash
# 使用默认配置开始训练
python train.py
```

**训练输出示例：**
```
Chinese License Plate Recognition — CRNN Training
  Device    : cuda
  Epochs    : 100  |  Batch: 128  |  LR: 0.001
  Results   : ./runs/train
  Weights   : ./runs/train/weights

  Train: 7848 samples  |  Val: 1963 samples

      Epoch    GPU_mem    ctc_loss   instances   img_size
        1/100     0.52G      2.1234          128      32x128
      ...
────────────────────────────────────────────────────────────
  Epoch 1/100  train_loss=2.1234  val_loss=1.8765  
  seq_acc=0.4521  char_acc=0.7834  lr=1.00e-03  time=00:05:32  *
────────────────────────────────────────────────────────────

  ✓ New best  seq_acc=0.4521  char_acc=0.7834  → saved best.pt
```

**训练产物说明：**
- `runs/train/args.yaml`: 本次训练的超参数配置
- `runs/train/results.csv`: 每轮的详细指标（CSV 格式）
- `runs/train/results.png`: 自动绘制的 Loss/Accuracy/LR 曲线
- `runs/train/weights/best.pt`: 验证集准确率最高的模型
- `runs/train/weights/last.pt`: 最近一次保存的检查点

#### 自定义训练参数

编辑 [`config.py`](config.py) 修改配置：

```python
@dataclass
class TrainConfig:
    batch_size: int = 128      # 批次大小
    epochs: int = 100          # 训练轮数
    lr: float = 0.001          # 初始学习率
    warmup_epochs: int = 5     # 预热轮数
    save_interval: int = 5     # 每隔多少轮保存一次检查点
    exp_name: str = 'train'    # 实验名称（自动递增）
```
这些配置项可以按需修改，以适应不同的实验需求。上传的训练模型也只是一个实验数据，不代表最终性能。
### 4. 模型推理

```bash
# 单张图片推理
python infer.py --image path/to/image.jpg

# 批量推理（评估整个目录）
python infer.py --dir path/to/images/

# 指定模型路径
python infer.py --image test.jpg \
                --model ./runs/train/weights/best.pt \
                --device cuda
```

**推理输出示例：**
```
Using device: cuda
Loading model from: ./runs/train/weights/best.pt
Loaded model from epoch 95
Validation accuracy: 0.9521

识别结果：京 A12345
置信度：0.9876
```

---

## 🏗️ 模型架构

### 改进的 CRNN (Convolutional Recurrent Neural Network)

本项目采用改进的 CRNN 架构，在传统 CRNN 基础上优化了 CNN 特征提取部分：

```
Input (1, 32, W)
    ↓
CNN Feature Extractor
    ├─ Conv(1→32) + BN + ReLU + MaxPool      # 第 1 层
    ├─ Conv(32→64) + BN + ReLU + MaxPool     # 第 2 层
    ├─ Conv(64→128) + BN + ReLU + MaxPool    # 第 3 层
    ├─ Conv(128→128) + BN + ReLU + MaxPool   # 第 4 层（非对称池化）
    └─ Conv(128→128) + BN + ReLU             # 第 5 层（额外卷积，替代自适应池化）
    ↓
Feature Sequence (W', 128)
    ↓
BiLSTM × 2 (hidden=128, bidirectional)
    ↓
FC Layer (256 → num_classes)
    ↓
Output (W', num_classes)
```

**网络结构特点：**

1. **CNN 特征提取器**:
   - 前四层使用标准卷积块（Conv + BN + ReLU + MaxPool）
   - 第四层采用非对称池化（宽度方向步长 1，高度方向步长 2）
   - 第五层使用额外的 2x2 卷积（stride=1）替代传统的自适应池化，更好地保留空间信息

2. **RNN 序列建模**:
   - 双层双向 LSTM，隐藏层维度 128
   - Dropout=0.2（当层数>1 时）

3. **分类层**:
   - 全连接层将双向 LSTM 输出映射到字符类别数

**与传统 CRNN 的区别：**
- 传统 CRNN 在 CNN 末端使用 AdaptiveAvgPool2d 强制高度为 1
- 本实现使用额外的卷积层（kernel_size=2, stride=1）进行特征融合，避免信息损失

### 支持的字符集

| 类别 | 字符 |
|------|------|
| **省份简称** | 京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙桂宁新藏 |
| **字母** | A-Z (不含 I、O) |
| **数字** | 0-9 |
| **特种字符** | 警学挂港澳使领险 |

**总计**: 78 个字符类别（含 blank）

---

## 📊 性能指标

在自建测试集上的典型表现（RTX 3080）：

| 模型 | 参数量 | 序列准确率 | 字符准确率 | 推理速度 |
|------|--------|-----------|-----------|---------|
| **CRNN-Small** | ~2.5M | 95.2% | 99.1% | ~5ms/img |
| **CRNN-Large** | ~8.5M | 97.1% | 99.5% | ~12ms/img |

*注：实际性能取决于数据集质量和硬件配置*

---

## 🔧 配置说明

### 核心配置文件

[`config.py`](config.py) 包含所有可配置项：

```python
# ── 模型配置 ──────────────────────────────────────
@dataclass
class ModelConfig:
    rnn_hidden: int = 256          # LSTM 隐藏层维度
    rnn_layers: int = 2            # LSTM 层数
    small: bool = True             # 轻量级模型
    img_height: int = 32           # 输入图像高度
    max_width: int = 128           # 最大图像宽度
    plate_chars: List[str] = [...] # 字符集

# ── 训练配置 ────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int = 128          # 批次大小
    epochs: int = 100              # 训练轮数
    lr: float = 0.001              # 初始学习率
    weight_decay: float = 1e-5     # 权重衰减
    warmup_epochs: int = 5         # 预热轮数
    num_workers: int = 4           # DataLoader 线程数
    save_dir: str = './runs'       # 结果保存目录
    exp_name: str = 'train'        # 实验名称
    log_interval: int = 10         # 日志打印间隔
    save_interval: int = 5         # 检查点保存间隔
    device: str = 'cuda'           # 计算设备

# ── 推理配置 ────────────────────────────────────
@dataclass
class InferConfig:
    model_path: str = './runs/train/weights/best.pt'
    device: str = 'cuda'
```

---

## 📈 训练可视化

### Results.png 曲线图

训练完成后自动生成 `results.png`，包含三个子图：

1. **CTC Loss 曲线**: 训练集 vs 验证集
2. **Validation Accuracy**: 序列准确率 vs 字符准确率
3. **Learning Rate 变化**: 余弦退火调度曲线

### Results.csv 数据导出

`results.csv` 包含每轮的完整指标，可用于自定义分析：

```csv
epoch,train/ctc_loss,val/ctc_loss,val/seq_acc,val/char_acc,lr
1,2.345123,1.987654,0.452100,0.783400,0.00100000
2,1.876543,1.654321,0.567800,0.823400,0.00098765
...
```

---

## 🛠️ 进阶用法

### 1. 断点续训

加载最近一次检查点继续训练：

```bash
python train.py --resume ./runs/train/weights/last.pt
```

### 2. 迁移学习

使用预训练权重微调新数据集：

```python
# 修改 config.py 中的字符集
model_cfg.plate_chars = ['新', '字', '符', '列', '表']

# 加载预训练权重（跳过 FC 层）
checkpoint = torch.load('pretrained.pt')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### 3. 模型导出 ONNX

```python
import torch

# 加载模型
model = CRNN(num_classes=78, small=True)
model.load_state_dict(torch.load('best.pt')['model_state_dict'])
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 1, 32, 128)
torch.onnx.export(model, dummy_input, "crnn.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 3: 'width'}})
```

---

## ⚠️ 注意事项

1. **输入图像要求**:
   - 必须为灰度图像（彩色会自动转换）
   - 双排车牌需预先处理为单排
   - 图像高度固定为 32，宽度自适应（最大 128）

2. **CTC Loss 机制**:
   - 索引 0 保留给 blank 字符
   - 真实标签长度不能超过输出序列长度

3. **显存优化**:
   - 如遇 OOM 错误，减小 `batch_size` 或启用 `num_workers=0`
   - 混合精度训练可进一步降低显存占用（待实现）

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

```bash
# Fork 项目
git clone https://github.com/eecn/chinese_license_plate_recognition.git

# 创建功能分支
git checkout -b feature/amazing-feature

# 提交更改
git commit -m 'Add amazing feature'

# 推送到分支
git push origin feature/amazing-feature
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

- 本项目基于 [CRNN](https://github.com/bgshih/crnn) 架构实现
- 感谢 [Ultralytics YOLO](https://github.com/ultralytics/yolov5) 的工程化设计启发
- 感谢 PyTorch 团队和开源社区的支持

---

## 📬 联系方式

如有问题请提 Issue，或通过以下方式联系：

- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/eecn/chinese_license_plate_recognition/issues)
- **仓库**: [https://github.com/eecn/chinese_license_plate_recognition](https://github.com/eecn/chinese_license_plate_recognition)

---

<div align="center">

**如果本项目对你有帮助，请给一个 ⭐️ Star！**

</div>