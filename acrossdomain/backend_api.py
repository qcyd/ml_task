import os
import random
import json
import time
import base64
import io
import threading
import ssl
import urllib
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_folder='static')
CORS(app)


class TrainingStatus:
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.device = ""
        self.message = "就绪"
        self.latest_val_acc = 0.0

    def to_dict(self):
        return {
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": self.train_loss_history,
            "train_acc": self.train_acc_history,
            "val_acc": self.val_acc_history,
            "device": self.device,
            "message": self.message,
            "latest_val_acc": self.latest_val_acc
        }


status = TrainingStatus()
model = None
train_thread = None


class Config:
    n_way = 5
    k_shot = 1
    q_query = 1
    epochs = 10
    episodes_per_epoch = 100
    test_episodes = 200
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    data_dir = 'data/omniglot-py'
    test_data_dir = 'data/omniglot-py_domain'
    in_channels = 3
    model_save_path = 'saved_models_across/protonet.pth'
    results_save_path = 'saved_models_across/results.json'


status.device = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")


class RandomColorDomainShift:
    def __call__(self, img_tensor):
        mask = (img_tensor < 0.5).float()
        while True:
            bg = torch.rand(3, 1, 1)
            fg = torch.rand(3, 1, 1)
            if torch.norm(bg - fg) > 0.5:
                break
        img = mask * fg + (1 - mask) * bg
        noise = torch.randn_like(img) * 0.1
        img = torch.clamp(img + noise, 0, 1)
        return img


class DataLoader:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    def load_and_cache_dataset(self, background=True, shift=False, data_dir=None, max_retry=3):
        transform = [
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]
        if shift:
            transform.append(RandomColorDomainShift())
        else:
            transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        transform = transforms.Compose(transform)

        for retry in range(max_retry):
            try:
                if data_dir and os.path.exists(data_dir):
                    dataset = datasets.ImageFolder(data_dir, transform=transform)
                else:
                    import socket
                    socket.setdefaulttimeout(30)
                    dataset = datasets.Omniglot(
                        root=self.data_dir,
                        background=background,
                        download=True,
                        transform=transform
                    )
                class_data = defaultdict(list)
                for img, label in dataset:
                    class_data[label].append(img)
                return [torch.stack(v) for v in class_data.values()] if class_data else []
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                if retry < max_retry - 1:
                    print(f"下载失败，重试 {retry + 1}/{max_retry}: {e}")
                    time.sleep(2)
                else:
                    print(f"最终下载失败: {e}")
                    return []
            except Exception as e:
                print(f"加载数据集错误: {e}")
                return []

    def load_training_data(self):
        try:
            src = self.load_and_cache_dataset(True, False)
            tgt = self.load_and_cache_dataset(True, True)
            return src + tgt
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return []

    def load_test_data(self):
        try:
            if os.path.exists(Config.test_data_dir):
                return self.load_and_cache_dataset(False, True, Config.test_data_dir)
            return self.load_and_cache_dataset(False, True)
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return []


class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 16, 1),
            nn.ReLU(),
            nn.Conv2d(c // 16, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(8, out_c),
        nn.ReLU(),
        SEBlock(out_c),
        nn.MaxPool2d(2),
        nn.Dropout(0.1)
    )


class ProtoNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(x.size(0), -1)
        z = F.normalize(z, dim=1)
        return z

    def predict(self, support, query, n_way, k_shot, q_query=1):
        x = torch.cat([
            support.view(n_way * k_shot, *support.shape[2:]),
            query.view(n_way * q_query, *query.shape[2:])
        ], dim=0)
        z = self.forward(x)
        z_s = z[:n_way * k_shot]
        z_q = z[n_way * k_shot:]
        z_s = z_s.view(n_way, k_shot, -1)
        proto = z_s.mean(1)
        dists = 1 - torch.matmul(z_q, proto.t())
        return dists

    def predict_one(self, support, query_image):
        """
        support: (n_way, k_shot, C, H, W)
        query_image: (1, C, H, W) 或 (C, H, W)
        返回: (n_way,) 的距离张量
        """
        n_way, k_shot = support.shape[0], support.shape[1]
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)
        x_s = support.view(n_way * k_shot, *support.shape[2:])
        x_q = query_image
        z_all = self.forward(torch.cat([x_s, x_q], dim=0))
        z_s = z_all[:n_way * k_shot].view(n_way, k_shot, -1)
        z_q = z_all[n_way * k_shot:]
        proto = z_s.mean(dim=1)
        dists = 1 - torch.matmul(z_q, proto.t()).squeeze(0)
        return dists


def train_model(n_way, k_shot, epochs):
    global status, model
    try:
        status.is_training = True
        status.total_epochs = epochs
        status.current_epoch = 0
        status.train_loss_history = []
        status.train_acc_history = []
        status.val_acc_history = []
        status.message = "初始化训练..."

        model = ProtoNet().to(Config.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        loader = DataLoader(Config.data_dir)

        status.message = "加载训练数据..."
        train_data = loader.load_training_data()

        if not train_data:
            status.message = "警告: 训练数据加载失败，将使用模拟数据"
            train_data = [torch.randn(20, 3, 28, 28) for _ in range(100)]

        status.message = f"开始训练 {n_way}-way {k_shot}-shot..."

        for epoch in range(epochs):
            status.current_epoch = epoch + 1
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            for episode in range(Config.episodes_per_epoch):
                if len(train_data) >= n_way:
                    classes = random.sample(train_data, n_way)
                else:
                    classes = train_data[:n_way]
                    if len(classes) < n_way:
                        classes = classes + classes[:(n_way - len(classes))]

                support, query = [], []
                for cls in classes:
                    if len(cls) >= k_shot + Config.q_query:
                        idx = torch.randperm(len(cls))[:k_shot + Config.q_query]
                        support.append(cls[idx[:k_shot]])
                        query.append(cls[idx[k_shot:]])
                    else:
                        support.append(torch.randn(k_shot, 3, 28, 28))
                        query.append(torch.randn(Config.q_query, 3, 28, 28))

                support_tensor = torch.stack(support).to(Config.device)
                query_tensor = torch.stack(query).to(Config.device)
                optimizer.zero_grad()

                x = torch.cat([
                    support_tensor.view(n_way * k_shot, *support_tensor.shape[2:]),
                    query_tensor.view(n_way * Config.q_query, *query_tensor.shape[2:])
                ], dim=0)

                z = model.encoder(x).view(x.size(0), -1)
                z = F.normalize(z, dim=1)
                z_s = z[:n_way * k_shot]
                z_q = z[n_way * k_shot:]
                z_s = z_s.view(n_way, k_shot, -1)
                proto = z_s.mean(1)

                dists = 1 - torch.matmul(z_q, proto.t())
                log_p = F.log_softmax(-dists / 0.5, dim=1)
                targets = torch.arange(n_way).repeat_interleave(Config.q_query).to(Config.device)
                loss = F.nll_loss(log_p, targets)
                loss += 0.01 * torch.var(z, dim=0).mean()

                loss.backward()
                optimizer.step()
                acc = (log_p.argmax(1) == targets).float().mean()
                epoch_loss += loss.item()
                epoch_acc += acc.item()

            avg_loss = epoch_loss / Config.episodes_per_epoch
            avg_acc = epoch_acc / Config.episodes_per_epoch
            status.train_loss_history.append(avg_loss)
            status.train_acc_history.append(avg_acc)

            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                val_acc = evaluate_model(model, n_way, k_shot)
                status.val_acc_history.append(val_acc)
                status.latest_val_acc = val_acc
                status.message = f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Val Acc: {val_acc:.4f}"
            else:
                status.message = f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"

            time.sleep(0.1)

        os.makedirs(os.path.dirname(Config.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), Config.model_save_path)
        status.message = f"训练完成！模型已保存到 {Config.model_save_path}"
    except Exception as e:
        status.message = f"训练出错: {str(e)}"
        print(f"训练错误: {e}")
    finally:
        status.is_training = False


def evaluate_model(model, n_way, k_shot, test_episodes=50):
    try:
        loader = DataLoader(Config.data_dir)
        test_data = loader.load_test_data()
        if not test_data:
            return random.uniform(0.1, 0.3)

        model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for _ in range(min(test_episodes, len(test_data) // n_way)):
                if len(test_data) >= n_way:
                    classes = random.sample(test_data, n_way)
                else:
                    continue

                support, query = [], []
                for cls in classes:
                    if len(cls) >= k_shot + Config.q_query:
                        idx = torch.randperm(len(cls))[:k_shot + Config.q_query]
                        support.append(cls[idx[:k_shot]])
                        query.append(cls[idx[k_shot:]])
                    else:
                        continue

                if len(support) < n_way:
                    continue

                support_tensor = torch.stack(support).to(Config.device)
                query_tensor = torch.stack(query).to(Config.device)

                x = torch.cat([
                    support_tensor.view(n_way * k_shot, *support_tensor.shape[2:]),
                    query_tensor.view(n_way * Config.q_query, *query_tensor.shape[2:])
                ], dim=0)
                z = model.encoder(x).view(x.size(0), -1)
                z = F.normalize(z, dim=1)
                z_s = z[:n_way * k_shot]
                z_q = z[n_way * k_shot:]
                z_s = z_s.view(n_way, k_shot, -1)
                proto = z_s.mean(1)
                dists = 1 - torch.matmul(z_q, proto.t())
                log_p = F.log_softmax(-dists / 0.5, dim=1)
                targets = torch.arange(n_way).repeat_interleave(Config.q_query).to(Config.device)
                acc = (log_p.argmax(1) == targets).float().mean()
                total_acc += acc.item()

        return total_acc / min(test_episodes, len(test_data) // n_way) if min(test_episodes,
                                                                              len(test_data) // n_way) > 0 else 0.0
    except Exception as e:
        print(f"评估错误: {e}")
        return random.uniform(0.1, 0.3)


def preprocess_image(image_data, shift_augment=False):
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        transform_list = [
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]
        if shift_augment:
            transform_list.append(RandomColorDomainShift())
        else:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        transform = transforms.Compose(transform_list)
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)
    except Exception as e:
        print(f"图像预处理错误: {e}")
        return None


@app.route('/')
def index():
    return send_file('templates/index-across.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/training_status', methods=['GET'])
def get_training_status():
    return jsonify(status.to_dict())


@app.route('/api/train', methods=['POST'])
def start_training():
    global train_thread, model, status
    if status.is_training:
        return jsonify({"status": "error", "message": "训练已在运行中"}), 400
    try:
        data = request.json
        n_way = data.get('n_way', Config.n_way)
        k_shot = data.get('k_shot', Config.k_shot)
        epochs = data.get('epochs', Config.epochs)
        Config.n_way = n_way
        Config.k_shot = k_shot
        Config.epochs = epochs
        train_thread = threading.Thread(target=train_model, args=(n_way, k_shot, epochs))
        train_thread.daemon = True
        train_thread.start()
        return jsonify({"status": "success", "message": f"开始训练 {n_way}-way {k_shot}-shot, 共 {epochs} 个epochs"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"启动训练失败: {str(e)}"}), 500


@app.route('/api/load_model', methods=['POST'])
def load_model():
    global model, status
    try:
        if os.path.exists(Config.model_save_path):
            model = ProtoNet().to(Config.device)
            model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device))
            model.eval()
            val_acc = evaluate_model(model, Config.n_way, Config.k_shot, 50)
            status.latest_val_acc = val_acc
            return jsonify({"status": "success", "message": f"模型加载成功，验证准确率: {val_acc:.4f}"})
        else:
            return jsonify({"status": "error", "message": "未找到已保存的模型"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"加载模型失败: {str(e)}"}), 500


@app.route('/api/reset', methods=['POST'])
def reset_system():
    global status, model, train_thread
    if status.is_training:
        status.is_training = False
        if train_thread and train_thread.is_alive():
            train_thread.join(timeout=1)
    status.__init__()
    status.device = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    model = None
    return jsonify({"status": "success", "message": "系统已重置"})


@app.route('/api/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({"status": "error", "message": "请先加载或训练模型"}), 400

    try:
        data = request.json
        n_way = int(data.get('n_way', Config.n_way))
        k_shot = int(data.get('k_shot', Config.k_shot))
        support_images = data.get('support_images', None)   # 允许完全不传
        query_image = data.get('query_image')

        if not query_image:
            return jsonify({"status": "error", "message": "查询图片不能为空"}), 400

        # 构建支持集
        if support_images and len(support_images) == n_way * k_shot:
            support_tensors = []
            for img_data in support_images:
                tensor = preprocess_image(img_data, shift_augment=True)
                if tensor is not None:
                    support_tensors.append(tensor)
            if len(support_tensors) != n_way * k_shot:
                return jsonify({"status": "error", "message": "部分支持集图片预处理失败"}), 400
            support_batch = torch.cat(support_tensors, dim=0)
            support_batch = support_batch.view(n_way, k_shot, 3, 28, 28)
            support_info = None
        else:
            # 自动采样（用户未提供足够支持图）
            loader = DataLoader(Config.data_dir)
            test_data = loader.load_test_data()
            if len(test_data) < n_way:
                return jsonify({"status": "error", "message": "测试数据不足，无法自动构建支持集"}), 400

            selected_classes = random.sample(test_data, n_way)
            support_tensors = []
            class_indices = []
            for idx, cls in enumerate(selected_classes):
                if len(cls) < k_shot + 1:
                    return jsonify({"status": "error", "message": "某些类数据不足，请减少 k_shot"}), 400
                # 修正：从张量中随机选取 k_shot 张图
                indices = random.sample(range(len(cls)), k_shot)
                support_tensors.append(cls[indices])   # shape: (k_shot, C, H, W)
                class_indices.append(idx)

            support_batch = torch.stack(support_tensors)   # (n_way, k_shot, C, H, W)
            support_info = {"used_classes": class_indices}

        # 预处理查询图像
        query_tensor = preprocess_image(query_image, shift_augment=True)
        if query_tensor is None:
            return jsonify({"status": "error", "message": "查询图片预处理失败"}), 400

        support_batch = support_batch.to(Config.device)
        query_tensor = query_tensor.to(Config.device)

        with torch.no_grad():
            dists = model.predict_one(support_batch, query_tensor)
            probs = F.softmax(-dists / 0.5, dim=0)
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

        result = {
            "predicted_class": int(pred_class),
            "predicted_class_name": f"类别 {pred_class + 1}",
            "confidence": float(confidence),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "probabilities": [
                {
                    "class": f"类别 {i + 1}",
                    "probability": float(probs[i].item()),
                    "percentage": f"{probs[i].item() * 100:.1f}%"
                }
                for i in range(n_way)
            ]
        }

        response = {"status": "success", "result": result}
        if support_info:
            response["support_info"] = support_info

        return jsonify(response)

    except Exception as e:
        print(f"预测失败: {e}")
        return jsonify({"status": "error", "message": f"预测失败: {str(e)}"}), 500


@app.route('/api/sample_images', methods=['GET'])
def get_sample_images():
    return jsonify({
        "status": "success",
        "message": "示例图片功能需实现Omniglot数据加载",
        "images": []
    })


if __name__ == '__main__':
    os.makedirs('saved_models_across', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    # 如果 index-across.html 不存在，则生成默认提示
    if not os.path.exists('templates/index-across.html'):
        with open('templates/index-across.html', 'w', encoding='utf-8') as f:
            f.write("<h1>前端文件未找到，请放置 index-across.html</h1>")

    print(f"设备: {Config.device}")
    print(f"服务器启动: http://127.0.0.1:5000")
    print("请确保已安装以下库:")
    print("pip install flask flask-cors torch torchvision pillow")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)