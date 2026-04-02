import os
import random
import sys

# 添加父目录到路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import train_cfg


# 数据预处理 划分训练集验证集 
# 数据格式(一行)：闽E335S8_1.jpg 闽E335S8  前面为图片路径，后面为车牌号
if __name__ == '__main__':

    base_dir = train_cfg.data_dir
    image_dir = os.path.join(base_dir, 'images')
    ratio = 0.8  # 训练集比例

    train_txt = os.path.join(base_dir, 'train.txt')
    val_txt = os.path.join(base_dir, 'val.txt')

    # 创建字典 key为车牌号 value为该车牌号所有图片路径列表
    plate_num_dict = {}  
    for img in os.listdir(image_dir):
        if img.endswith('.jpg'):
            img_path = os.path.join(image_dir, img)
            plate_num = os.path.splitext(img)[0].split('_')[0]
            if plate_num not in plate_num_dict:
                plate_num_dict[plate_num] = []
            plate_num_dict[plate_num].append(img_path)

    # 划分训练集和验证集
    train_list = []
    val_list = []
    for plate_num, imgs in plate_num_dict.items():
        img_len = len(imgs)
        val_num = int(img_len * (1-ratio)) # 不足一个 val_num为0
        
        # 只有一个数据随机划分
        if img_len == 1:
            if random.random() < 0.5:
                train_list.append(imgs[0])
            else:
                val_list.append(imgs[0])
        elif val_num >= 1: # 有多个数据 val_num
            if random.random() < ratio:
                train_list.extend(imgs)
            else:
                val_list.extend(imgs)
        else: # 有多个数据 val_num为0 也就是不足一个 保证至少包含一个
            val_idx = random.randint(0, img_len-1)
            val_list.append(imgs[val_idx])
            train_list.extend([img for idx, img in enumerate(imgs) if idx != val_idx])

    with open(train_txt, 'w', encoding='utf-8') as trainf:
        for img in train_list:
            img_name = os.path.basename(img).split('_')[0]
            trainf.write(img + ' ' + img_name + '\n')

    with open(val_txt, 'w', encoding='utf-8') as valf:
        for img in val_list:
            img_name = os.path.basename(img).split('_')[0]
            valf.write(img + ' ' + img_name + '\n')
