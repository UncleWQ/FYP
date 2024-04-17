import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time

# β值调度生成函数
def generate_beta_schedule(num_steps=1000, start=0.0001, end=0.02):
    return np.linspace(start, end, num_steps)

# 扩散过程添加噪声
def diffusion_step(x, beta, device):
    std = torch.sqrt(torch.tensor(beta, device=device))
    noise = torch.randn_like(x) * std
    return x + noise, noise

# U-Net 模型定义
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),  # 修正为192, 因为 u1 (64) + e2 (128)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),  # 修正为96, 因为 u2 (32) + e1 (64)
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        u1 = self.upconv1(p2)
        c1 = torch.cat((u1, e2), 1)
        d1 = self.decoder1(c1)

        u2 = self.upconv2(d1)
        c2 = torch.cat((u2, e1), 1)
        d2 = self.decoder2(c2)

        return d2



# 自定义数据集类
class NPYDataset(Dataset):
    def __init__(self, directory, betas, device):
        self.directory = directory
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        self.betas = betas
        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data_path = self.filenames[idx]
        data = np.load(data_path).astype(np.float32)  # 确保数据类型为 float32
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
        noisy_data, _ = diffusion_step(data_tensor, self.betas[-1], self.device)  # 使用最后一个 beta 值添加噪声
        return noisy_data, data_tensor


# 训练函数
def train(model, dataloader, epochs, optimizer, criterion, device):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for noisy_data, original_data in dataloader:
            optimizer.zero_grad()
            predicted_noise = model(noisy_data)
            loss = criterion(predicted_noise, noisy_data - original_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    return train_losses

# 逆扩散过程
def reverse_diffusion(model, dataloader, output_folder, betas, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model.eval()
    with torch.no_grad():
        for i, (noisy_data, _) in enumerate(dataloader):
            data = noisy_data.to(device)
            for beta in reversed(betas):  # Start from the last noise and reverse
                predicted_noise = model(data)
                std = torch.sqrt(torch.tensor(beta, device=device))
                data = (data - predicted_noise) / (1 + std)  # Approximate reverse diffusion

            for j, img in enumerate(data):  # 遍历批次中的每个图像
                img = img.squeeze().cpu().numpy()  # 移除通道维度，准备保存
                if img.ndim > 2:
                    img = img.squeeze()  # 确保图像是二维的
                filename = os.path.basename(dataloader.dataset.filenames[i * dataloader.batch_size + j])
                new_filename = filename.replace('.npy', '-01.npy')
                output_path = os.path.join(output_folder, new_filename)
                np.save(output_path, img.astype(np.float64))  # 保存为 float64 类型


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name = torch.cuda.get_device_name()
print('Using device '+ name + ' to train the model.')
betas = generate_beta_schedule()
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 加载数据并训练
dataset = NPYDataset('/tmp/FYP_Projects/12_DTweighted/train', betas, device)
dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
print("Data loaded!")
train_losses = train(model, dataloader, 10, optimizer, criterion, device)
torch.save(model.state_dict(),'/tmp/FYP_Projects/diffusion_model.pt')
print("Model saved!")

# 可视化损失
plt.plot(train_losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig("Diffusion_loss.jpg")

# 逆扩散过程
print("Begin to generate!")
start_time = time.time()
reverse_diffusion(model, dataloader, '/tmp/FYP_Projects/Diffusion', betas, device)
elapsed = (time.time()-start_time) / 60
print(f'Task finished! cost_time={elapsed:.3f} min')
