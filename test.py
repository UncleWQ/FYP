import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn.functional import relu
from torch.optim.lr_scheduler import StepLR

# 检查CUDA是否可用，选择正确的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.time_embedding = nn.Linear(1, 16)
        self.down1 = nn.Sequential(nn.Conv2d(1 + 16, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.final = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = relu(self.time_embedding(t))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_emb], dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up2(x3)
        x = self.up3(x + x2)
        x = self.final(x + x1)
        return x

# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        if not self.files:
            raise RuntimeError("No data files found in specified directory.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        if img.max() == img.min():  # Check for constant image
            img = torch.zeros_like(img)  # Use a zero tensor if the image is constant
        else:
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        return img, os.path.basename(self.files[idx])

# 扩散和逆扩散步骤
# 扩散步骤
def diffusion_step(x, beta):
    noise = torch.randn_like(x).to(device)
    return torch.sqrt(torch.clamp(1 - beta, min=1e-8)) * x + torch.sqrt(torch.clamp(beta, min=1e-8)) * noise

def reverse_diffusion_step(x_t, t, model, beta):
    pred_noise = model(x_t, t)
    return (x_t - torch.sqrt(beta) * pred_noise) / torch.sqrt(1 - beta)

# 训练函数
def train(model, data_loader, epochs=10, beta_schedule=np.linspace(0.01, 0.1, 1000)):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for batch, _ in data_loader:
            x = batch.to(device)
            t = torch.rand(x.size(0), 1).to(device)
            beta = torch.tensor(beta_schedule[np.random.randint(0, len(beta_schedule))], device=device)
            x_t = diffusion_step(x, beta)
            pred_noise = model(x_t, t)
            true_noise = x - diffusion_step(x, beta)
            loss = loss_fn(pred_noise, true_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 采样函数和保存
def generate_and_save_samples(model, data_loader, output_dir='train_new', beta_schedule=np.linspace(0.01, 0.1, 1000)):
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timesteps = 1000
    with torch.no_grad():
        for batch, filenames in data_loader:
            for x, filename in zip(batch, filenames):
                x = x.unsqueeze(0).to(device)
                x_t = torch.randn_like(x)
                for t in reversed(range(timesteps)):
                    time_step = torch.tensor([[t / timesteps]], device=device)
                    beta = torch.tensor(beta_schedule[t], device=device)
                    x_t = reverse_diffusion_step(x_t, time_step, model, beta)
                sample_np = x_t.cpu().numpy().squeeze()
                variant_filename = os.path.basename(filename).replace('.npy', '-01.npy')
                np.save(os.path.join(output_dir, variant_filename), sample_np.astype(np.float64))

# 实例化模型和数据加载
model = UNet().to(device)
dataset = ImageDataset('train')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

print("data loaded!")
# 训练模型
train(model, dataloader)

# 生成样本并保存
generate_and_save_samples(model, dataloader)

print("Task finished!")
