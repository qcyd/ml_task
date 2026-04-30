import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def random_color():
    """生成随机RGB颜色"""
    return np.array([
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ], dtype=np.uint8)


def add_noise(image, noise_level=30):
    """
    添加随机噪声
    """
    noise = np.random.randint(0, noise_level, image.shape, dtype=np.uint8)
    noisy_img = image.astype(np.int16) + noise.astype(np.int16)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def process_image(img_path, save_path):
    """
    处理单张图片：
    前景+背景随机颜色 + 噪声
    """
    # 读取灰度图
    img = Image.open(img_path).convert('L')
    img_np = np.array(img)

    h, w = img_np.shape

    # 👉 Omniglot：白底黑字
    mask = img_np < 128  # True表示字符区域

    # 随机颜色
    bg_color = random_color()
    fg_color = random_color()

    # 创建彩色图像
    color_img = np.ones((h, w, 3), dtype=np.uint8)

    # 背景填充
    color_img[:] = bg_color

    # 前景填充
    color_img[mask] = fg_color

    # 添加噪声
    color_img = add_noise(color_img, noise_level=40)

    # 保存
    Image.fromarray(color_img).save(save_path)


def process_dataset(input_root, output_root):
    """
    遍历数据集并处理
    """
    all_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.png'):
                all_files.append(os.path.join(root, file))

    for input_path in tqdm(all_files, desc="Processing"):
        # 构造输出路径
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, relative_path)

        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        process_image(input_path, output_path)





if __name__ == "__main__":
    # 原始Omniglot路径
    input_dir = "data/omniglot-py"

    # 输出路径
    output_dir = "data/omniglot-py_domain"

    process_dataset(input_dir, output_dir)

    print("✅ 处理完成！新数据集已保存到:", output_dir)