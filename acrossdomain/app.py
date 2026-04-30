import json
import os
import random
import threading
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response

# ===========================
# 配置
# ===========================
class Config:
    n_way = 5
    k_shot = 1
    q_query = 5
    epochs = 60
    episodes_per_epoch = 100
    test_episodes = 100
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data/omniglot-py'
    img_size = 28


# ===========================
# 温和颜色扰动
# ===========================
class MildColorShift:
    def __call__(self, x):
        color = torch.rand(3, 1, 1) * 0.2 + 0.9
        x = x * color
        noise = torch.randn_like(x) * 0.02
        return torch.clamp(x + noise, 0, 1)


# ===========================
# 模型
# ===========================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.fc = nn.Linear(512, 128)

    def forward(self, s, q, n_way, k_shot, q_query):
        x = torch.cat([
            s.view(n_way * k_shot, *s.shape[2:]),
            q.view(n_way * q_query, *q.shape[2:])
        ], dim=0)

        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        z = self.fc(x)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        z_s = z[:n_way * k_shot]
        z_q = z[n_way * k_shot:]

        z_s = z_s.view(n_way, k_shot, -1)
        proto = z_s.mean(1)

        dists = torch.cdist(z_q, proto, p=2)
        log_p = F.log_softmax(-dists, dim=1)

        target = torch.arange(n_way).repeat_interleave(q_query).to(z.device)
        loss = F.nll_loss(log_p, target)
        acc = (log_p.argmax(1) == target).float().mean()

        return loss, acc

    def predict(self, support_imgs, query_img):
        """
        support_imgs: list of Tensors [ (k,) 3x28x28 ] for each class
        query_img: Tensor 3x28x28
        返回: 预测类别索引和各类别距离
        """
        n_way = len(support_imgs)
        k_shot = support_imgs[0].size(0)  # 假设每类支持数量相同

        # 构建 batch：支持集在前，查询在后
        support_batch = torch.cat(support_imgs, dim=0)  # (n_way*k_shot, 3, 28, 28)
        query_batch = query_img.unsqueeze(0)            # (1, 3, 28, 28)

        x = torch.cat([support_batch, query_batch], dim=0)
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        z = self.fc(x)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        z_s = z[:n_way * k_shot].view(n_way, k_shot, -1)
        z_q = z[-1:]                                    # 1x128

        proto = z_s.mean(1)                             # n_way x 128
        dists = torch.cdist(z_q, proto, p=2).squeeze()  # n_way

        pred = dists.argmin().item()
        return pred, dists.tolist()


# ===========================
# 数据加载器（仅用于训练）
# ===========================
class Loader:
    def __init__(self, root):
        self.root = root

    def load(self, shift=False):
        transform = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            MildColorShift() if shift else transforms.Lambda(lambda x: x)
        ])

        root = os.path.join(self.root, 'images_background')
        data = defaultdict(list)

        for a in os.listdir(root):
            for c in os.listdir(os.path.join(root, a)):
                path = os.path.join(root, a, c)
                if not os.path.isdir(path):
                    continue
                imgs = []
                for f in os.listdir(path):
                    img = Image.open(os.path.join(path, f)).convert('L')
                    imgs.append(transform(img))
                data[f"{a}/{c}"] = torch.stack(imgs)

        return list(data.values())


# ===========================
# 全局状态
# ===========================
app = Flask(__name__)

cfg = Config()
model = ProtoNet().to(cfg.device)
opt = optim.Adam(model.parameters(), lr=cfg.lr)

# 训练进度（SSE用）
progress_lock = threading.Lock()
training_status = {
    "running": False,
    "epoch": 0,
    "max_epochs": cfg.epochs,
    "loss": [],
    "acc": [],
    "test_result": None
}

# 数据集（全局加载一次，节省内存）
try:
    loader = Loader(cfg.data_dir)
    src_data = loader.load(shift=False)
    tgt_data = loader.load(shift=True)
    data_loaded = True
except Exception as e:
    print(f"数据集加载失败: {e}")
    src_data, tgt_data = [], []
    data_loaded = False


# ===========================
# 图片预处理（用于单张推理）
# ===========================
inference_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 转三通道，无颜色扰动
])


def preprocess_image(file_storage):
    """将 Flask FileStorage 转为模型可接受的 Tensor"""
    img = Image.open(file_storage).convert('L')
    t = inference_transform(img)
    return t


# ===========================
# 训练线程
# ===========================
def train_thread():
    global training_status, model, opt

    with progress_lock:
        training_status["running"] = True
        training_status["loss"] = []
        training_status["acc"] = []
        training_status["test_result"] = None

    # 采样函数
    def sample():
        data = tgt_data if random.random() < 0.5 else src_data
        cls = random.sample(data, cfg.n_way)
        s, q = [], []
        for c in cls:
            idx = torch.randperm(len(c))[:cfg.k_shot + cfg.q_query]
            s.append(c[idx[:cfg.k_shot]])
            q.append(c[idx[cfg.k_shot:]])
        return torch.stack(s).to(cfg.device), torch.stack(q).to(cfg.device)

    for ep in range(cfg.epochs):
        model.train()
        ls_total, ac_total = 0, 0

        for _ in range(cfg.episodes_per_epoch):
            s, q = sample()
            opt.zero_grad()
            loss, acc = model(s, q, cfg.n_way, cfg.k_shot, cfg.q_query)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls_total += loss.item()
            ac_total += acc.item()

        avg_loss = ls_total / cfg.episodes_per_epoch
        avg_acc = ac_total / cfg.episodes_per_epoch

        with progress_lock:
            training_status["epoch"] = ep + 1
            training_status["loss"].append(avg_loss)
            training_status["acc"].append(avg_acc)

        time.sleep(0.1)  # 给 SSE 一点时间推送

    # 训练结束，执行一次测试
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for _ in range(cfg.test_episodes):
            cls = random.sample(tgt_data, cfg.n_way)
            s, q = [], []
            for c in cls:
                idx = torch.randperm(len(c))[:cfg.k_shot + cfg.q_query]
                s.append(c[idx[:cfg.k_shot]])
                q.append(c[idx[cfg.k_shot:]])
            s = torch.stack(s).to(cfg.device)
            q = torch.stack(q).to(cfg.device)
            _, a = model(s, q, cfg.n_way, cfg.k_shot, cfg.q_query)
            test_acc += a.item()
    test_acc_avg = test_acc / cfg.test_episodes

    with progress_lock:
        training_status["running"] = False
        training_status["test_result"] = round(test_acc_avg, 4)


# ===========================
# Flask 路由
# ===========================
@app.route('/')
def index():
    return render_template('index-across.html', data_loaded=data_loaded)


@app.route('/start_train', methods=['POST'])
def start_train():
    if not data_loaded:
        return jsonify({"error": "Omniglot 数据集未加载，请检查路径"}), 400
    with progress_lock:
        if training_status["running"]:
            return jsonify({"error": "训练已在进行中"}), 400
    thread = threading.Thread(target=train_thread)
    thread.start()
    return jsonify({"message": "训练已开始"})


@app.route('/progress')
def progress():
    def generate():
        while True:
            with progress_lock:
                data = {
                    "epoch": training_status["epoch"],
                    "max_epochs": training_status["max_epochs"],
                    "loss": training_status["loss"][-1] if training_status["loss"] else 0,
                    "acc": training_status["acc"][-1] if training_status["acc"] else 0,
                    "history_loss": training_status["loss"],
                    "history_acc": training_status["acc"],
                    "running": training_status["running"],
                    "test_result": training_status["test_result"]
                }
            yield f"data:{json.dumps(data)}\n\n"
            if not data["running"] and data["test_result"] is not None:
                # 最后再发一次完整结果后关闭
                time.sleep(1)
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')


@app.route('/single_test', methods=['POST'])
def single_test():
    """
    前端应发送：
    - 支持集图片组：class_0_img, class_1_img, ... 每个可多文件
    - 查询图片：query_img
    - 类别名（可选）：class_names[]  如果不提供则用数字索引
    返回：预测类别名（或索引）及所有距离
    """
    if model is None:
        return jsonify({"error": "模型尚未初始化"}), 500

    # 收集所有类别
    class_keys = []
    for key in request.files:
        if key.startswith('class_'):
            # 格式: class_0_img 或 class_1_img
            # 取出类别索引
            prefix = key.split('_img')[0]  # class_0
            if prefix not in class_keys:
                class_keys.append(prefix)
    class_keys.sort(key=lambda x: int(x.split('_')[1]))

    if len(class_keys) < 2:
        return jsonify({"error": "至少需要两个类别"}), 400

    # 读取每个类别的支持图片
    support_tensors = []
    class_names = request.form.getlist('class_names')  # 可选
    for i, ck in enumerate(class_keys):
        # ck 是 "class_0" 等形式
        files = request.files.getlist(f"{ck}_img")
        if not files:
            return jsonify({"error": f"类别 {ck} 未上传图片"}), 400
        imgs = []
        for f in files:
            try:
                imgs.append(preprocess_image(f))
            except Exception as e:
                return jsonify({"error": f"处理图片失败: {str(e)}"}), 400
        support_tensors.append(torch.stack(imgs).to(cfg.device))

    # 读取查询图片
    query_file = request.files.get('query_img')
    if query_file is None:
        return jsonify({"error": "未上传查询图片"}), 400
    try:
        query_tensor = preprocess_image(query_file).to(cfg.device)
    except Exception as e:
        return jsonify({"error": f"查询图片处理失败: {str(e)}"}), 400

    # 模型预测
    model.eval()
    with torch.no_grad():
        pred_idx, dists = model.predict(support_tensors, query_tensor)

    # 确定预测类别名称
    if class_names and len(class_names) == len(support_tensors):
        pred_name = class_names[pred_idx]
    else:
        pred_name = f"类别 {pred_idx}"

    return jsonify({
        "prediction": pred_name,
        "pred_index": pred_idx,
        "distances": {f"类别 {i}" if not class_names else class_names[i]: round(d, 4) for i, d in enumerate(dists)}
    })


if __name__ == '__main__':
    app.run(debug=True, threaded=True)