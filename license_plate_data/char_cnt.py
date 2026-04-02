import os
import sys

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import train_cfg, model_cfg


# 遍历文件夹中的文件名 使用_分割 提取_前面部分的内容 分割字符 统计一共有多少字符 以及每个字符的数量
def count_chars(folder_path):
    char_count = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
            file_name = file_name.split('_')[0]
            for char in file_name:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
    return char_count


if __name__ == '__main__':
    # 从配置文件获取图像目录
    folder_path = os.path.join(train_cfg.data_dir, 'images')
    
    print(f"Scanning directory: {folder_path}")
    char_count = count_chars(folder_path)
    
    print("\n字符统计结果:")
    print("-" * 40)
    for char, count in sorted(char_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {char}: {count}")
    
    print("\n" + "-" * 40)
    print(f"总字符种类数: {len(char_count)}")
    
    # 检查是否有配置文件中没有的字符
    config_chars = set(model_cfg.plate_chars)
    dataset_chars = set(char_count.keys())
    missing_chars = dataset_chars - config_chars
    
    if missing_chars:
        print(f"\n警告: 以下字符在数据集中但不在配置文件中:")
        for char in sorted(missing_chars):
            print(f"  '{char}'")
        print("请更新 config.py 中的 plate_chars 列表")
    else:
        print("✓ 所有字符都已在配置文件中定义")
    
    # 以列表形式打印字符 以便复制
    print("\n字符列表 (可直接复制到配置):")
    print("[", end='')
    for char in char_count.keys():
        print(f"'{char}',", end='')
    print("]")
