import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset


# ==========================================
# 1. 超参数设置
# ==========================================
class Config:
    n_way = 5  # N-way: 每次任务（Episode）选取的类别数量
    k_shot = 1  # K-shot: 每个类别的支持集（Support Set）样本数
    q_query = 5  # 每个类别的查询集（Query Set）样本数

    epochs = 10  # 训练轮数 (作业中为了快速演示设为10，实际可设为更高)
    episodes_per_epoch = 100  # 每轮训练包含的Episode数量
    test_episodes = 200  # 测试时的Episode数量

    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))

    data_dir = './data'
    in_channels = 3  # 跨域升级：使用RGB 3通道图像输入


# ==========================================
# 2. 数据集预处理与Episode采样器
# ==========================================
class RandomColorDomainShift:
    """
    自定义数据增强：为灰度图像添加随机背景和笔画颜色，模拟跨域(Domain Shift)场景。
    """

    def __call__(self, img_tensor):
        # img_tensor 形状为 [1, H, W], 值域 [0, 1]
        # 翻转像素：Omniglot ToTensor后背景接近1，笔画接近0。翻转使笔画为1，背景为0
        mask = 1.0 - img_tensor

        # 随机生成背景颜色和笔画颜色 (RGB 3通道)
        bg_color = torch.rand(3, 1, 1)
        stroke_color = torch.rand(3, 1, 1)

        # 合成彩色图像: mask区域填入笔画颜色，其余填入背景色
        colored_img = mask * stroke_color + (1.0 - mask) * bg_color
        return colored_img


def load_and_cache_dataset(background=True, apply_domain_shift=False):
    """
    加载Omniglot数据集并将按类别对图像进行分组。
    background=True 表示训练集(背景集)，False 表示测试集(评估集)。
    apply_domain_shift=True 表示应用随机颜色变换，模拟目标域。
    """
    domain_name = "跨域(彩色)" if apply_domain_shift else "源域(灰度)"
    print(f"正在加载 {'训练集(Background)' if background else '测试集(Evaluation)'} [{domain_name}] 数据...")

    # 调整图像大小为 28x28，并转换为张量
    transform_list = [
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ]

    if apply_domain_shift:
        transform_list.append(RandomColorDomainShift())
    else:
        # 如果不模拟跨域，也将其扩展为3通道以匹配网络输入
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    transform = transforms.Compose(transform_list)

    dataset = datasets.Omniglot(root=Config.data_dir, background=background,
                                download=True, transform=transform)

    # 按类别缓存数据，加快采样速度
    class_data = defaultdict(list)
    for img, label in dataset:
        class_data[label].append(img)

    # 转换为列表形式，每个元素是一个类别的所有图片张量 [Num_classes, Num_images_per_class, C, H, W]
    cached_data = [torch.stack(images) for images in class_data.values()]
    return cached_data


def generate_episode(cached_data, n_way, k_shot, q_query):
    """
    生成一个Episode（任务），包含支持集和查询集
    """
    # 随机选择 n_way 个类别
    selected_classes = random.sample(cached_data, n_way)

    support_set = []
    query_set = []

    for class_images in selected_classes:
        # 每个类别随机选择 k_shot + q_query 个样本
        # 注意：Omniglot每个类别有20个样本
        indices = torch.randperm(len(class_images))[:k_shot + q_query]
        samples = class_images[indices]

        support_set.append(samples[:k_shot])
        query_set.append(samples[k_shot:])

    # 形状: (n_way, k_shot, C, H, W) 和 (n_way, q_query, C, H, W)
    return torch.stack(support_set).to(Config.device), torch.stack(query_set).to(Config.device)


# ==========================================
# 3. 原型网络模型 (Prototypical Network)
# ==========================================
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        # 使用经典的 4 层卷积块作为特征提取器 (Encoder)
        # 修改: 第一层输入通道改为 Config.in_channels (3)
        self.encoder = nn.Sequential(
            self.conv_block(Config.in_channels, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 【核心跨域优化】将 BatchNorm 替换为 InstanceNorm
            # InstanceNorm 能在单一图片样本内部消除全局风格（如背景颜色）的干扰
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 每次池化将特征图尺寸减半
        )

    def euclidean_dist(self, x, y):
        """
        计算平方欧式距离
        x: 查询集特征 (N*Q, D)
        y: 原型特征 (N, D)
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def forward(self, support, query, n_way, k_shot, q_query):
        """
        前向传播，计算损失和准确率
        """
        # 合并 batch 维度以便通过卷积层
        # [n_way * k_shot, C, H, W]
        x = torch.cat([support.view(n_way * k_shot, *support.shape[2:]),
                       query.view(n_way * q_query, *query.shape[2:])], dim=0)

        # 提取特征
        z = self.encoder(x)
        z = z.view(z.size(0), -1)  # 展平为向量

        # 分离支持集和查询集的特征
        z_support = z[:n_way * k_shot]
        z_query = z[n_way * k_shot:]

        # 将支持集按类别重塑并计算类的原型 (Prototype)
        # 形状: [n_way, k_shot, D] -> [n_way, D] (在k_shot维度上取均值)
        z_support = z_support.view(n_way, k_shot, -1)
        prototypes = z_support.mean(dim=1)

        # 计算查询集与各类原型之间的欧式距离
        dists = self.euclidean_dist(z_query, prototypes)

        # 使用负距离作为 logits 进行 Softmax 分类
        log_p_y = F.log_softmax(-dists, dim=1)

        # 生成查询集的真实标签 (每个类 q_query 个，总共 n_way * q_query 个)
        # 例如 n_way=3, q=2 时标签为: [0,0, 1,1, 2,2]
        targets = torch.arange(n_way).repeat_interleave(q_query).to(Config.device)

        # 计算 NLL 损失
        loss = F.nll_loss(log_p_y, targets)

        # 计算预测准确率
        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, targets).float().mean()

        return loss, acc


# ==========================================
# 4. 训练和评估流程
# ==========================================
def train(model, optimizer, train_data):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for i in range(Config.episodes_per_epoch):
        # 采样一个任务(Episode)
        support, query = generate_episode(train_data, Config.n_way, Config.k_shot, Config.q_query)

        optimizer.zero_grad()
        loss, acc = model(support, query, Config.n_way, Config.k_shot, Config.q_query)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / Config.episodes_per_epoch, total_acc / Config.episodes_per_epoch


def evaluate(model, test_data):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for i in range(Config.test_episodes):
            support, query = generate_episode(test_data, Config.n_way, Config.k_shot, Config.q_query)
            loss, acc = model(support, query, Config.n_way, Config.k_shot, Config.q_query)

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / Config.test_episodes, total_acc / Config.test_episodes


# ==========================================
# 5. 主函数
# ==========================================
if __name__ == '__main__':
    print(f"使用设备: {Config.device}")
    print(f"任务设置: {Config.n_way}-Way {Config.k_shot}-Shot")

    # 准备数据：域随机化策略
    # 训练时：我们混合使用原图和随机上色的图进行训练，迫使模型学习形状特征
    train_data_source = load_and_cache_dataset(background=True, apply_domain_shift=False)
    train_data_shifted = load_and_cache_dataset(background=True, apply_domain_shift=True)
    train_data = train_data_source + train_data_shifted

    # 测试集全部使用域变换 (随机彩色背景) 的数据，评估跨域能力
    test_data_shifted = load_and_cache_dataset(background=False, apply_domain_shift=True)

    # 初始化模型与优化器
    model = ProtoNet().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    print("\n开始跨域训练 (Domain Randomization)...")
    for epoch in range(1, Config.epochs + 1):
        train_loss, train_acc = train(model, optimizer, train_data)

        print(f"Epoch {epoch:2d}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

    print("\n开始跨域测试评估 (目标域：随机颜色背景)...")
    test_loss, test_acc = evaluate(model, test_data_shifted)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    print("\n跨域小样本识别训练与测试完成！")