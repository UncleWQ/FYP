import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.time_embedding = nn.Linear(1, 16)
        self.down1 = nn.Sequential(nn.Conv2d(1 + 16, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32))
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(64))
        self.down3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(128))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(64))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32))
        self.final = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t).relu()
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_emb], dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up3(x3) + x2
        x = self.up2(x) + x1
        x = self.final(x)
        return x

def diffusion_step(x, beta):
    std = torch.sqrt(beta)
    noise = torch.randn_like(x).to(x.device)
    return x * torch.sqrt(1 - beta) + std * noise

def reverse_diffusion_step(x_noisy, t, model, beta):
    pred_noise = model(x_noisy, t)
    std = torch.sqrt(beta)
    return (x_noisy - std * pred_noise) / torch.sqrt(1 - beta)

def generate_beta_schedule(num_steps=1000, start=0.0001, end=0.02):
    return np.linspace(start, end, num_steps)

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        if not self.files:
            raise RuntimeError("No data files found in specified directory.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载图像文件
        img = np.load(self.files[idx])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # 应用百分位归一化
        img = self.percentile_normalize(img)
        return img, os.path.basename(self.files[idx])

    def percentile_normalize(self, img):
        # 计算1%和99%百分位数
        lower = np.percentile(img.numpy(), 1)
        upper = np.percentile(img.numpy(), 99)
        # 归一化图像
        img = torch.clamp((img - lower) / (upper - lower), 0, 1)
        return img
    

def generate_and_save_samples(model, data_loader, output_dir='train_new', beta_schedule = generate_beta_schedule(num_steps=1000, start=0.0001, end=0.02)):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timesteps = 1000  # 考虑根据需要调整这个值
    with torch.no_grad():
        for batch, filenames in data_loader:
            for x, filename in zip(batch, filenames):
                x = x.unsqueeze(0).to(device)
                x_t = torch.randn_like(x)  # 初始噪声
                for t in reversed(range(timesteps)):
                    time_step = torch.tensor([[t / timesteps]], device=device)
                    beta = torch.tensor(beta_schedule[t], device=device)
                    x_t = reverse_diffusion_step(x_t, time_step, model, beta)
                sample_np = x_t.cpu().numpy().squeeze()
                variant_filename = os.path.basename(filename).replace('.npy', '-01.npy')
                np.save(os.path.join(output_dir, variant_filename), sample_np.astype(np.float64))


def train(model, data_loader, epochs=50, beta_schedule=None):
    model.train()
    name = torch.cuda.get_device_name()
    print('Using device '+ name + ' to train the model.')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss_fn = nn.MSELoss()
    losses = []

    if beta_schedule is None:
        raise ValueError("Beta schedule is not provided.")

    for epoch in range(epochs):
        start = time.time()
        epoch_loss = 0
        for batch, _ in data_loader:
            x = batch.to(device)
            t = torch.rand(x.size(0), 1).to(device)
            beta_index = np.random.randint(0, len(beta_schedule))
            beta = torch.tensor(beta_schedule[beta_index], device = device)
            x_t = diffusion_step(x, beta)
            pred_noise = model(x_t, t)
            true_noise = x - diffusion_step(x, beta)
            loss = loss_fn(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        elapsed = (time.time()-start) / 60
        print(f'Training epoch={epoch+1} \t cost_time={elapsed:.3f} min \t loss={avg_loss:.6f}')

    return losses

# 实例化模型和数据加载
model = UNet().to(device)
dataset = ImageDataset('/tmp/FYP_Projects/12_DTweighted/train')
dataloader = DataLoader(dataset, batch_size=100, shuffle=True,num_workers=8)
print("Dataset loaded!")

# 训练模型
beta_schedule = generate_beta_schedule(num_steps=1000, start=0.0001, end=0.02)
losses = train(model, dataloader, beta_schedule = beta_schedule)
torch.save(model.state_dict(),'/tmp/FYP_Projects/diffusion_model_new.pt')
print("Model saved!")

plt.figure(figsize=(12,6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig("/tmp/FYP_Projects/training_loss_new.jpg")

# 生成样本并保存
start_time = time.time()
print(start_time)
generate_and_save_samples(model, dataloader)
gen_time = (time.time()-start_time) / 60
print(f'Task finished! cost_time={gen_time:.3f} min')
