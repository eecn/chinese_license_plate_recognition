import os
import argparse
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from model import CRNN
from utils import ctc_decode, char2idx
from config import model_cfg, infer_cfg

# 从配置文件加载默认参数
DEFAULT_PLATE_CHARS = model_cfg.plate_chars
DEFAULT_IMG_HEIGHT = model_cfg.img_height
DEFAULT_MAX_WIDTH = model_cfg.max_width
DEFAULT_MODEL_PATH = infer_cfg.model_path
DEFAULT_DEVICE = infer_cfg.device


def preprocess(img, img_height=DEFAULT_IMG_HEIGHT, max_width=DEFAULT_MAX_WIDTH):
    """
    预处理图像
    
    Args:
        img: 输入图像 (numpy array)
        img_height: 目标图像高度
        max_width: 最大图像宽度
        
    Returns:
        预处理后的图像张量 (1, 1, H, W)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    ratio = img_height / h
    new_w = int(w * ratio)
    if new_w > max_width:
        new_w = max_width
    img = cv2.resize(img, (new_w, img_height))
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
    return torch.from_numpy(img).float()


def recognize(model, img_tensor, idx2char, device):
    """
    识别单张图像
    
    Args:
        model: CRNN 模型
        img_tensor: 预处理后的图像张量
        idx2char: 索引到字符的映射
        device: 计算设备
        
    Returns:
        (识别文本, 置信度)
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))  # (1, T, C)
    text, conf = ctc_decode(logits[0], idx2char)
    return text, conf


def load_model(model_path, num_classes, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重路径
        num_classes: 字符类别数
        device: 计算设备
        
    Returns:
        加载好的模型
    """
    model = CRNN(num_classes=num_classes, small=model_cfg.small)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 支持两种格式的模型加载
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def inference_single(image_path, model, idx2char, device, img_height=32, max_width=128):
    """
    对单张图片进行推理
    
    Args:
        image_path: 图像路径
        model: 加载好的模型
        idx2char: 索引到字符的映射
        device: 计算设备
        img_height: 图像高度
        max_width: 最大宽度
        
    Returns:
        (识别结果, 置信度)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return None, 0.0
    
    img_tensor = preprocess(img, img_height, max_width)
    text, conf = recognize(model, img_tensor, idx2char, device)
    return text, conf


def inference_batch(image_dir, model, idx2char, device, img_height=32, max_width=128):
    """
    对目录中的所有图片进行批量推理
    
    Args:
        image_dir: 图像目录
        model: 加载好的模型
        idx2char: 索引到字符的映射
        device: 计算设备
        img_height: 图像高度
        max_width: 最大宽度
        
    Returns:
        结果列表 [(image_path, text, confidence), ...]
    """
    image_paths = glob(os.path.join(image_dir, '*.jpg')) + \
                  glob(os.path.join(image_dir, '*.png')) + \
                  glob(os.path.join(image_dir, '*.jpeg'))
    
    results = []
    correct = 0
    total = 0
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    for img_path in tqdm(image_paths, desc="Processing"):
        text, conf = inference_single(img_path, model, idx2char, device, img_height, max_width)
        if text is not None:
            results.append((img_path, text, conf))
            
            # 尝试从文件名提取真实标签（用于评估）
            basename = os.path.basename(img_path)
            gt = basename.split('_')[0].split('.')[0]
            if gt and all(c in DEFAULT_PLATE_CHARS for c in gt):
                total += 1
                if text == gt:
                    correct += 1
                status = "✓" if text == gt else "✗"
                print(f"{status} {basename}: {text} (conf: {conf:.4f})")
            else:
                print(f"  {basename}: {text} (conf: {conf:.4f})")
    
    if total > 0:
        accuracy = correct / total
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Chinese License Plate Recognition Inference')
    parser.add_argument('--image', '-i', type=str, help='Single image path')
    parser.add_argument('--dir', '-d', type=str, help='Directory containing images')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_PATH,
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--img-height', type=int, default=DEFAULT_IMG_HEIGHT,
                        help='Input image height')
    parser.add_argument('--max-width', type=int, default=DEFAULT_MAX_WIDTH,
                        help='Max input image width')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备字符映射
    _, idx2char, num_classes = char2idx(DEFAULT_PLATE_CHARS)
    
    # 加载模型
    print(f"Loading model from: {args.model}")
    model = load_model(args.model, num_classes, device)
    
    # 执行推理
    if args.image:
        text, conf = inference_single(
            args.image, model, idx2char, device, 
            args.img_height, args.max_width
        )
        print(f"\n识别结果: {text}")
        print(f"置信度: {conf:.4f}")
    elif args.dir:
        inference_batch(
            args.dir, model, idx2char, device,
            args.img_height, args.max_width
        )
    else:
        # 默认使用 test.jpg
        default_image = './test.jpg'
        if os.path.exists(default_image):
            text, conf = inference_single(
                default_image, model, idx2char, device,
                args.img_height, args.max_width
            )
            print(f"\n识别结果: {text}")
            print(f"置信度: {conf:.4f}")
        else:
            print("Please provide --image or --dir argument, or place a test.jpg in the current directory")


if __name__ == '__main__':
    main()