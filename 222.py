import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 超参数设置
# ==========================================
class Config:
    n_way = 5  # N-way: 每次任务（Episode）选取的类别数量
    k_shot = 1  # K-shot: 每个类别的支持集（Support Set）样本数
    q_query = 5  # 每个类别的查询集（Query Set）样本数

    epochs = 10  # 训练轮数
    episodes_per_epoch = 100  # 每轮训练包含的Episode数量
    test_episodes = 200  # 测试时的Episode数量

    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))

    data_dir = 'archive'

    # 可视化相关参数
    save_plots = True  # 是否保存可视化图像
    plot_dir = 'plots'  # 保存图像的目录


# ==========================================
# 2. 数据集预处理与Episode采样器
# ==========================================
def load_and_cache_dataset(background=True):
    """
    加载Omniglot数据集并将按类别对图像进行分组。
    background=True 表示训练集(背景集)，False 表示测试集(评估集)。
    """
    print(f"正在加载 {'训练集(Background)' if background else '测试集(Evaluation)'} 数据...")

    # 调整图像大小为 28x28，并转换为张量
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    dataset = datasets.Omniglot(root=Config.data_dir, background=background,
                                download=True, transform=transform)

    # 按类别缓存数据，加快采样速度
    class_data = defaultdict(list)
    for img, label in dataset:
        class_data[label].append(img)

    # 转换为列表形式，每个元素是一个类别的所有图片张量
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
        self.encoder = nn.Sequential(
            self.conv_block(1, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
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
        x = torch.cat([support.view(n_way * k_shot, *support.shape[2:]),
                       query.view(n_way * q_query, *query.shape[2:])], dim=0)

        # 提取特征
        z = self.encoder(x)
        z = z.view(z.size(0), -1)  # 展平为向量

        # 分离支持集和查询集的特征
        z_support = z[:n_way * k_shot]
        z_query = z[n_way * k_shot:]

        # 将支持集按类别重塑并计算类的原型 (Prototype)
        z_support = z_support.view(n_way, k_shot, -1)
        prototypes = z_support.mean(dim=1)

        # 计算查询集与各类原型之间的欧式距离
        dists = self.euclidean_dist(z_query, prototypes)

        # 使用负距离作为 logits 进行 Softmax 分类
        log_p_y = F.log_softmax(-dists, dim=1)

        # 生成查询集的真实标签
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
def train(model, optimizer, train_data, epoch_history):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    # 记录每个episode的损失和准确率，用于分析训练过程
    episode_losses = []
    episode_accs = []

    for i in range(Config.episodes_per_epoch):
        # 采样一个任务(Episode)
        support, query = generate_episode(train_data, Config.n_way, Config.k_shot, Config.q_query)

        optimizer.zero_grad()
        loss, acc = model(support, query, Config.n_way, Config.k_shot, Config.q_query)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

        # 记录每个episode的指标
        episode_losses.append(loss.item())
        episode_accs.append(acc.item())

    avg_loss = total_loss / Config.episodes_per_epoch
    avg_acc = total_acc / Config.episodes_per_epoch

    # 将每个epoch的训练过程细节保存下来
    epoch_history['episode_losses'].append(episode_losses)
    epoch_history['episode_accs'].append(episode_accs)
    epoch_history['avg_losses'].append(avg_loss)
    epoch_history['avg_accs'].append(avg_acc)

    return avg_loss, avg_acc


def evaluate(model, test_data):
    """
    在测试集上评估模型性能
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    # 记录所有测试episode的指标
    test_losses = []
    test_accs = []

    with torch.no_grad():
        for i in range(Config.test_episodes):
            support, query = generate_episode(test_data, Config.n_way, Config.k_shot, Config.q_query)
            loss, acc = model(support, query, Config.n_way, Config.k_shot, Config.q_query)

            total_loss += loss.item()
            total_acc += acc.item()

            test_losses.append(loss.item())
            test_accs.append(acc.item())

    avg_loss = total_loss / Config.test_episodes
    avg_acc = total_acc / Config.test_episodes

    return avg_loss, avg_acc, test_losses, test_accs


# ==========================================
# 5. 可视化函数
# ==========================================
def create_plot_directory():
    """创建保存可视化图像的目录"""
    if Config.save_plots and not os.path.exists(Config.plot_dir):
        os.makedirs(Config.plot_dir)


def plot_training_curves(epoch_history, test_results=None):
    """
    绘制训练过程中的损失和准确率曲线
    """
    epochs = list(range(1, len(epoch_history['avg_losses']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 平均损失和准确率曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, epoch_history['avg_losses'], 'b-', linewidth=2, label='训练损失')
    ax1.set_xlabel('训练轮数 (Epoch)')
    ax1.set_ylabel('损失 (Loss)')
    ax1.set_title(f'{Config.n_way}-Way {Config.k_shot}-Shot 训练损失曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc * 100 for acc in epoch_history['avg_accs']], 'g-', linewidth=2, label='训练准确率')
    ax2.set_xlabel('训练轮数 (Epoch)')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title(f'{Config.n_way}-Way {Config.k_shot}-Shot 训练准确率曲线')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 2. 第一个epoch和最后一个epoch的episode损失分布对比
    ax3 = axes[1, 0]
    if len(epoch_history['episode_losses']) >= 1:
        first_epoch_losses = epoch_history['episode_losses'][0]
        last_epoch_losses = epoch_history['episode_losses'][-1]

        bins = np.linspace(0, max(max(first_epoch_losses), max(last_epoch_losses)) + 0.5, 30)
        ax3.hist(first_epoch_losses, bins=bins, alpha=0.5, label=f'Epoch 1 (平均: {np.mean(first_epoch_losses):.3f})')
        ax3.hist(last_epoch_losses, bins=bins, alpha=0.5,
                 label=f'Epoch {len(epochs)} (平均: {np.mean(last_epoch_losses):.3f})')
        ax3.set_xlabel('损失值')
        ax3.set_ylabel('频数')
        ax3.set_title('训练损失分布变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 3. 第一个epoch和最后一个epoch的episode准确率分布对比
    ax4 = axes[1, 1]
    if len(epoch_history['episode_accs']) >= 1:
        first_epoch_accs = [acc * 100 for acc in epoch_history['episode_accs'][0]]
        last_epoch_accs = [acc * 100 for acc in epoch_history['episode_accs'][-1]]

        bins = np.linspace(0, 100, 21)
        ax4.hist(first_epoch_accs, bins=bins, alpha=0.5, label=f'Epoch 1 (平均: {np.mean(first_epoch_accs):.1f}%)')
        ax4.hist(last_epoch_accs, bins=bins, alpha=0.5,
                 label=f'Epoch {len(epochs)} (平均: {np.mean(last_epoch_accs):.1f}%)')
        ax4.set_xlabel('准确率 (%)')
        ax4.set_ylabel('频数')
        ax4.set_title('训练准确率分布变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if Config.save_plots:
        plt.savefig(os.path.join(Config.plot_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_test_results(test_loss, test_acc, test_losses, test_accs):
    """
    绘制测试结果可视化
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 测试损失分布
    ax1 = axes[0]
    n, bins, patches = ax1.hist(test_losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=test_loss, color='red', linestyle='--', linewidth=2,
                label=f'平均损失: {test_loss:.3f}')
    ax1.set_xlabel('测试损失')
    ax1.set_ylabel('频数')
    ax1.set_title(f'测试损失分布 ({Config.test_episodes}个测试任务)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加文本标注
    ax1.text(0.05, 0.95,
             f'最小损失: {min(test_losses):.3f}\n最大损失: {max(test_losses):.3f}\n标准差: {np.std(test_losses):.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 测试准确率分布
    ax2 = axes[1]
    test_accs_percent = [acc * 100 for acc in test_accs]
    n, bins, patches = ax2.hist(test_accs_percent, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(x=test_acc * 100, color='red', linestyle='--', linewidth=2,
                label=f'平均准确率: {test_acc * 100:.2f}%')
    ax2.set_xlabel('测试准确率 (%)')
    ax2.set_ylabel('频数')
    ax2.set_title(f'测试准确率分布 ({Config.test_episodes}个测试任务)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 添加文本标注
    ax2.text(0.05, 0.95,
             f'最低准确率: {min(test_accs_percent):.1f}%\n最高准确率: {max(test_accs_percent):.1f}%\n标准差: {np.std(test_accs_percent):.1f}%',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if Config.save_plots:
        plt.savefig(os.path.join(Config.plot_dir, 'test_results.png'), dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_combined_summary(epoch_history, test_loss, test_acc):
    """
    绘制训练和测试的综合总结图
    """
    epochs = list(range(1, len(epoch_history['avg_losses']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练和验证曲线（这里验证用测试代替）
    ax1 = axes[0, 0]
    train_loss_line, = ax1.plot(epochs, epoch_history['avg_losses'], 'b-', linewidth=2, label='训练损失')
    ax1.set_xlabel('训练轮数 (Epoch)')
    ax1.set_ylabel('损失 (Loss)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('训练损失曲线')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    train_acc_line, = ax1_twin.plot(epochs, [acc * 100 for acc in epoch_history['avg_accs']],
                                    'g-', linewidth=2, label='训练准确率')
    ax1_twin.set_ylabel('准确率 (%)', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')

    # 合并图例
    lines = [train_loss_line, train_acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 2. 最终测试结果柱状图
    ax2 = axes[0, 1]
    categories = ['训练最终轮', '测试']
    loss_values = [epoch_history['avg_losses'][-1], test_loss]
    acc_values = [epoch_history['avg_accs'][-1] * 100, test_acc * 100]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, loss_values, width, label='损失', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('阶段')
    ax2.set_ylabel('损失值', color='darkred')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_title('最终损失对比')

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width / 2, acc_values, width, label='准确率', color='lightgreen', alpha=0.8)
    ax2_twin.set_ylabel('准确率 (%)', color='darkgreen')
    ax2_twin.tick_params(axis='y', labelcolor='darkgreen')

    # 在柱子上添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax2.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.01,
                 f'{bar1.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        ax2_twin.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.5,
                      f'{bar2.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. 训练准确率随epoch变化的热力图风格
    ax3 = axes[1, 0]
    episode_acc_matrix = np.array(epoch_history['episode_accs']).T * 100

    im = ax3.imshow(episode_acc_matrix, aspect='auto', cmap='YlGn', interpolation='nearest')
    ax3.set_xlabel('训练轮数 (Epoch)')
    ax3.set_ylabel('Episode索引')
    ax3.set_title('每个Episode训练准确率热力图')
    plt.colorbar(im, ax=ax3, label='准确率 (%)')

    # 4. 模型性能总结
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = (
        f'模型训练总结\n\n'
        f'任务设置: {Config.n_way}-Way {Config.k_shot}-Shot\n'
        f'训练轮数: {Config.epochs}\n'
        f'每轮Episode数: {Config.episodes_per_epoch}\n'
        f'测试Episode数: {Config.test_episodes}\n\n'
        f'训练结果:\n'
        f'• 初始损失: {epoch_history["avg_losses"][0]:.3f}\n'
        f'• 最终损失: {epoch_history["avg_losses"][-1]:.3f}\n'
        f'• 损失降低: {(epoch_history["avg_losses"][0] - epoch_history["avg_losses"][-1]):.3f}\n'
        f'• 初始准确率: {epoch_history["avg_accs"][0] * 100:.1f}%\n'
        f'• 最终准确率: {epoch_history["avg_accs"][-1] * 100:.1f}%\n'
        f'• 准确率提升: {(epoch_history["avg_accs"][-1] - epoch_history["avg_accs"][0]) * 100:.1f}%\n\n'
        f'测试结果:\n'
        f'• 测试损失: {test_loss:.3f}\n'
        f'• 测试准确率: {test_acc * 100:.2f}%\n'
        f'• 过拟合程度: {(epoch_history["avg_accs"][-1] - test_acc) * 100:.2f}%'
    )

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if Config.save_plots:
        plt.savefig(os.path.join(Config.plot_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')

    plt.show()
    return fig


# ==========================================
# 6. 主函数
# ==========================================
if __name__ == '__main__':
    print(f"使用设备: {Config.device}")
    print(f"任务设置: {Config.n_way}-Way {Config.k_shot}-Shot")
    print(f"训练轮数: {Config.epochs}, 每轮Episode数: {Config.episodes_per_epoch}")
    print(f"测试Episode数: {Config.test_episodes}")
    print("=" * 60)

    # 创建保存图像的目录
    create_plot_directory()

    # 准备数据
    train_data = load_and_cache_dataset(background=True)
    test_data = load_and_cache_dataset(background=False)

    # 初始化模型与优化器
    model = ProtoNet().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 初始化训练历史记录
    epoch_history = {
        'avg_losses': [],  # 每个epoch的平均损失
        'avg_accs': [],  # 每个epoch的平均准确率
        'episode_losses': [],  # 每个epoch中每个episode的损失
        'episode_accs': []  # 每个epoch中每个episode的准确率
    }

    print("\n开始训练...")
    print("-" * 60)

    # 训练循环
    for epoch in range(1, Config.epochs + 1):
        train_loss, train_acc = train(model, optimizer, train_data, epoch_history)

        # 打印当前epoch的训练结果
        print(f"Epoch {epoch:2d}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

        # 每5个epoch打印一次详细统计
        if epoch % 5 == 0 or epoch == 1:
            epoch_losses = epoch_history['episode_losses'][-1]
            epoch_accs = epoch_history['episode_accs'][-1]
            print(f"  Episode损失范围: [{min(epoch_losses):.3f}, {max(epoch_losses):.3f}]")
            print(f"  Episode准确率范围: [{min(epoch_accs) * 100:.1f}%, {max(epoch_accs) * 100:.1f}%]")
            print("-" * 60)

    print("\n开始测试评估...")
    test_loss, test_acc, test_losses, test_accs = evaluate(model, test_data)
    print(f"测试结果: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}%")
    print("=" * 60)

    # 计算并显示训练过程中的改进
    initial_loss = epoch_history['avg_losses'][0]
    final_loss = epoch_history['avg_losses'][-1]
    initial_acc = epoch_history['avg_accs'][0]
    final_acc = epoch_history['avg_accs'][-1]

    print(f"训练过程改进总结:")
    print(f"  • 损失降低: {initial_loss:.4f} → {final_loss:.4f} (降低 {initial_loss - final_loss:.4f})")
    print(
        f"  • 准确率提升: {initial_acc * 100:.2f}% → {final_acc * 100:.2f}% (提升 {(final_acc - initial_acc) * 100:.2f}%)")
    print(f"  • 过拟合程度: {(final_acc - test_acc) * 100:.2f}% (训练准确率 - 测试准确率)")

    # 可视化训练过程和测试结果
    print("\n生成可视化图表...")

    # 1. 绘制训练曲线
    fig1 = plot_training_curves(epoch_history)

    # 2. 绘制测试结果分布
    fig2 = plot_test_results(test_loss, test_acc, test_losses, test_accs)

    # 3. 绘制综合总结
    fig3 = plot_combined_summary(epoch_history, test_loss, test_acc)

    print("\n训练与测试完成！")
    if Config.save_plots:
        print(f"可视化图表已保存到 '{Config.plot_dir}' 目录")
    print("祝你作业顺利！")
