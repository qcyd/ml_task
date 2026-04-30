import os
import random
import json
import time
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import base64
import io
import threading

app = Flask(__name__)
CORS(app)

# 全局变量用于存储训练进度
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': [],
    'train_acc': [],
    'test_loss': None,
    'test_acc': None,
    'message': ''
}

# 全局模型和优化器
model = None
optimizer = None
train_data = None
test_data = None


# ==========================================
# 配置类
# ==========================================
class Config:
    n_way = 5
    k_shot = 1
    q_query = 5
    epochs = 10
    episodes_per_epoch = 100
    test_episodes = 200
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    data_dir = 'archive'
    model_save_path = 'saved_models/prototypical_net.pth'
    results_save_path = 'saved_models/training_results.json'


# ==========================================
# 数据集处理函数
# ==========================================
def load_and_cache_dataset(background=True):
    """加载Omniglot数据集"""
    print(f"正在加载 {'训练集' if background else '测试集'} 数据...")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    dataset = datasets.Omniglot(root=Config.data_dir, background=background,
                                download=True, transform=transform)

    class_data = defaultdict(list)
    for img, label in dataset:
        class_data[label].append(img)

    cached_data = [torch.stack(images) for images in class_data.values()]
    return cached_data


def generate_episode(cached_data, n_way, k_shot, q_query):
    """生成一个训练任务（Episode）"""
    selected_classes = random.sample(cached_data, n_way)
    support_set = []
    query_set = []

    for class_images in selected_classes:
        indices = torch.randperm(len(class_images))[:k_shot + q_query]
        samples = class_images[indices]
        support_set.append(samples[:k_shot])
        query_set.append(samples[k_shot:])

    return torch.stack(support_set).to(Config.device), torch.stack(query_set).to(Config.device)


# ==========================================
# 原型网络模型
# ==========================================
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
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
            nn.MaxPool2d(2)
        )

    def euclidean_dist(self, x, y):
        """计算欧式距离"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def forward(self, support, query, n_way, k_shot, q_query):
        """前向传播"""
        x = torch.cat([support.view(n_way * k_shot, *support.shape[2:]),
                       query.view(n_way * q_query, *query.shape[2:])], dim=0)

        z = self.encoder(x)
        z = z.view(z.size(0), -1)

        z_support = z[:n_way * k_shot]
        z_query = z[n_way * k_shot:]

        z_support = z_support.view(n_way, k_shot, -1)
        prototypes = z_support.mean(dim=1)

        dists = self.euclidean_dist(z_query, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        targets = torch.arange(n_way).repeat_interleave(q_query).to(Config.device)
        loss = F.nll_loss(log_p_y, targets)

        _, y_hat = log_p_y.max(1)
        acc = torch.eq(y_hat, targets).float().mean()

        return loss, acc, prototypes, z_query

    def predict(self, support_images, query_image, n_way, k_shot):
        """使用支持集和查询图像进行预测"""
        with torch.no_grad():
            # 确保支持集形状正确
            if support_images.dim() == 4:
                # 形状为 (n_way * k_shot, 1, 28, 28)
                support = support_images.view(n_way, k_shot, 1, 28, 28)
            else:
                # 如果已经是5维，直接使用
                support = support_images

            # 确保查询图像形状正确
            if query_image.dim() == 3:
                # 形状为 (1, 28, 28)，添加批次维度
                query = query_image.unsqueeze(0)
            else:
                query = query_image

            # 合并并通过编码器
            x = torch.cat([support.view(n_way * k_shot, 1, 28, 28), query], dim=0)
            z = self.encoder(x)
            z = z.view(z.size(0), -1)

            # 分离特征
            z_support = z[:n_way * k_shot]
            z_query = z[n_way * k_shot:]

            # 计算原型
            z_support = z_support.view(n_way, k_shot, -1)
            prototypes = z_support.mean(dim=1)

            # 计算距离和概率
            dists = self.euclidean_dist(z_query, prototypes)
            probabilities = F.softmax(-dists, dim=1)

            return probabilities.squeeze().cpu().numpy()


# ==========================================
# 训练函数
# ==========================================
def train_epoch(model, optimizer, train_data, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for i in range(Config.episodes_per_epoch):
        support, query = generate_episode(train_data, Config.n_way, Config.k_shot, Config.q_query)
        optimizer.zero_grad()
        loss, acc, _, _ = model(support, query, Config.n_way, Config.k_shot, Config.q_query)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    avg_loss = total_loss / Config.episodes_per_epoch
    avg_acc = total_acc / Config.episodes_per_epoch

    # 更新训练状态
    training_status['train_loss'].append(avg_loss)
    training_status['train_acc'].append(avg_acc)
    training_status['current_epoch'] = epoch

    return avg_loss, avg_acc


def evaluate_model(model, test_data):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for i in range(Config.test_episodes):
            support, query = generate_episode(test_data, Config.n_way, Config.k_shot, Config.q_query)
            loss, acc, _, _ = model(support, query, Config.n_way, Config.k_shot, Config.q_query)
            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / Config.test_episodes, total_acc / Config.test_episodes


# ==========================================
# 训练线程函数
# ==========================================
def training_thread():
    """在单独线程中运行训练"""
    global model, optimizer, train_data, test_data, training_status

    try:
        training_status['is_training'] = True
        training_status['train_loss'] = []
        training_status['train_acc'] = []
        training_status['message'] = '开始加载数据...'

        # 加载数据
        train_data = load_and_cache_dataset(background=True)
        test_data = load_and_cache_dataset(background=False)

        # 初始化模型
        model = ProtoNet().to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

        training_status['message'] = f'开始训练，设备: {Config.device}'
        training_status['total_epochs'] = Config.epochs

        # 训练循环
        for epoch in range(1, Config.epochs + 1):
            train_loss, train_acc = train_epoch(model, optimizer, train_data, epoch)

            training_status['message'] = (
                f'Epoch {epoch}/{Config.epochs} | '
                f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc * 100:.2f}%'
            )
            time.sleep(0.5)  # 让前端有时间更新

        # 评估模型
        training_status['message'] = '开始评估模型...'
        test_loss, test_acc = evaluate_model(model, test_data)
        training_status['test_loss'] = test_loss
        training_status['test_acc'] = test_acc

        # 保存模型
        os.makedirs('saved_models', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'n_way': Config.n_way,
                'k_shot': Config.k_shot,
                'q_query': Config.q_query
            }
        }, Config.model_save_path)

        # 保存训练结果
        results = {
            'train_loss': training_status['train_loss'],
            'train_acc': training_status['train_acc'],
            'test_loss': training_status['test_loss'],
            'test_acc': training_status['test_acc'],
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_way': Config.n_way,
                'k_shot': Config.k_shot,
                'epochs': Config.epochs
            }
        }

        with open(Config.results_save_path, 'w') as f:
            json.dump(results, f, indent=2)

        training_status['message'] = (
            f'训练完成！测试准确率: {test_acc * 100:.2f}% | '
            f'模型已保存到 {Config.model_save_path}'
        )

    except Exception as e:
        training_status['message'] = f'训练出错: {str(e)}'
    finally:
        training_status['is_training'] = False


# ==========================================
# Flask API 路由
# ==========================================
@app.route('/api/train', methods=['POST'])
def start_training():
    """开始训练"""
    if training_status['is_training']:
        return jsonify({'status': 'error', 'message': '训练已在进行中'})

    # 解析请求参数
    data = request.json or {}
    Config.n_way = data.get('n_way', Config.n_way)
    Config.k_shot = data.get('k_shot', Config.k_shot)
    Config.epochs = data.get('epochs', Config.epochs)

    # 启动训练线程
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'success',
        'message': '训练已开始',
        'config': {
            'n_way': Config.n_way,
            'k_shot': Config.k_shot,
            'epochs': Config.epochs
        }
    })


@app.route('/api/training_status', methods=['GET'])
def get_training_status():
    """获取训练状态"""
    return jsonify(training_status)


@app.route('/api/predict', methods=['POST'])
def predict():
    """使用模型进行预测"""
    global model

    if model is None:
        return jsonify({'status': 'error', 'message': '模型未加载，请先训练'})

    try:
        data = request.json
        if not data or 'support_images' not in data or 'query_image' not in data:
            return jsonify({'status': 'error', 'message': '缺少必要参数'})

        n_way = int(data.get('n_way', Config.n_way))
        k_shot = int(data.get('k_shot', Config.k_shot))

        # 检查支持集图像数量
        support_images = data['support_images']
        if len(support_images) != n_way * k_shot:
            return jsonify({
                'status': 'error',
                'message': f'支持集图像数量错误。期望 {n_way * k_shot} 张（{n_way} 个类别，每个类别 {k_shot} 张），实际收到 {len(support_images)} 张'
            })

        # 转换支持集图像
        support_tensors = []
        for img_base64 in support_images:
            img_data = base64.b64decode(img_base64.split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('L')
            img = transforms.Resize((28, 28))(img)
            img_tensor = transforms.ToTensor()(img)
            support_tensors.append(img_tensor)

        support_tensor = torch.stack(support_tensors).to(Config.device)

        # 转换查询图像
        query_data = base64.b64decode(data['query_image'].split(',')[1])
        query_img = Image.open(io.BytesIO(query_data)).convert('L')
        query_img = transforms.Resize((28, 28))(query_img)
        query_tensor = transforms.ToTensor()(query_img).to(Config.device)

        # 进行预测
        probabilities = model.predict(
            support_tensor,
            query_tensor,
            n_way,
            k_shot
        )

        # 准备结果
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'class': f'类别 {i + 1}',
                'probability': float(prob),
                'percentage': f'{prob * 100:.2f}%'
            })

        # 获取预测类别
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return jsonify({
            'status': 'success',
            'predicted_class': predicted_class,
            'predicted_class_name': f'类别 {predicted_class + 1}',
            'confidence': confidence,
            'confidence_percentage': f'{confidence * 100:.2f}%',
            'probabilities': results
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/load_model', methods=['POST'])
def load_saved_model():
    """加载已保存的模型"""
    global model

    try:
        if not os.path.exists(Config.model_save_path):
            return jsonify({'status': 'error', 'message': '未找到保存的模型'})

        checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
        model = ProtoNet().to(Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 加载训练结果
        if os.path.exists(Config.results_save_path):
            with open(Config.results_save_path, 'r') as f:
                results = json.load(f)
            training_status.update({
                'train_loss': results.get('train_loss', []),
                'train_acc': results.get('train_acc', []),
                'test_loss': results.get('test_loss'),
                'test_acc': results.get('test_acc')
            })

        return jsonify({'status': 'success', 'message': '模型加载成功'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/sample_images', methods=['GET'])
def get_sample_images():
    """获取示例图像用于测试"""
    try:
        # 加载数据集
        if train_data is None:
            data = load_and_cache_dataset(background=True)
        else:
            data = train_data

        # 随机选择一些图像
        selected_class = random.choice(data)
        samples = random.sample(range(len(selected_class)), min(10, len(selected_class)))

        # 转换为base64
        image_data = []
        for idx in samples[:5]:  # 只返回5张
            img_tensor = selected_class[idx]
            img = transforms.ToPILImage()(img_tensor)

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            image_data.append(f"data:image/png;base64,{img_str}")

        return jsonify({
            'status': 'success',
            'images': image_data
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/')
def index():
    """主页面"""
    return send_file('templates/index.html')


# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    print(f"原型网络系统启动中...")
    print(f"设备: {Config.device}")
    print(f"任务设置: {Config.n_way}-Way {Config.k_shot}-Shot")
    print(f"访问 http://localhost:5000 使用网页界面")

    app.run(debug=True, port=5000)
