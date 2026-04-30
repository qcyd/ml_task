import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import defaultdict
from PIL import Image



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



class MildColorShift:
    def __call__(self, x):
        # 直接对整图做轻微颜色扰动（不做mask）
        color = torch.rand(3, 1, 1) * 0.2 + 0.9
        x = x * color

        noise = torch.randn_like(x) * 0.02
        return torch.clamp(x + noise, 0, 1)



class Loader:
    def __init__(self, root):
        self.root = root

    def load(self, shift=False):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
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

        # 防止 NaN
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



class Trainer:
    def __init__(self):
        self.cfg = Config()
        self.model = ProtoNet().to(self.cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        loader = Loader(self.cfg.data_dir)
        self.src = loader.load(shift=False)
        self.tgt = loader.load(shift=True)

    def sample(self):
        # 不混域
        data = self.tgt if random.random() < 0.5 else self.src

        cls = random.sample(data, self.cfg.n_way)

        s, q = [], []
        for c in cls:
            idx = torch.randperm(len(c))[:self.cfg.k_shot + self.cfg.q_query]
            s.append(c[idx[:self.cfg.k_shot]])
            q.append(c[idx[self.cfg.k_shot:]])

        return torch.stack(s).to(self.cfg.device), torch.stack(q).to(self.cfg.device)

    def train(self):
        for ep in range(self.cfg.epochs):
            self.model.train()
            ls, ac = 0, 0

            for _ in range(self.cfg.episodes_per_epoch):
                s, q = self.sample()

                self.opt.zero_grad()
                loss, acc = self.model(s, q,
                                       self.cfg.n_way,
                                       self.cfg.k_shot,
                                       self.cfg.q_query)

                loss.backward()


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.opt.step()

                ls += loss.item()
                ac += acc.item()

            print(f"Epoch {ep+1} | Loss {ls/100:.4f} | Acc {ac/100:.4f}")

        self.test()

    def test(self):
        self.model.eval()
        acc = 0

        with torch.no_grad():
            for _ in range(self.cfg.test_episodes):
                cls = random.sample(self.tgt, self.cfg.n_way)

                s, q = [], []
                for c in cls:
                    idx = torch.randperm(len(c))[:self.cfg.k_shot + self.cfg.q_query]
                    s.append(c[idx[:self.cfg.k_shot]])
                    q.append(c[idx[self.cfg.k_shot:]])

                s = torch.stack(s).to(self.cfg.device)
                q = torch.stack(q).to(self.cfg.device)

                _, a = self.model(s, q,
                                  self.cfg.n_way,
                                  self.cfg.k_shot,
                                  self.cfg.q_query)

                acc += a.item()

        print("\n Test Acc:", acc / self.cfg.test_episodes)


if __name__ == "__main__":
    Trainer().train()