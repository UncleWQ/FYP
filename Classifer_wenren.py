###Data load by wenren
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 自定义数据集类
class NPYDataset(Dataset):
    def __init__(self, root):
        self.data, self.labels = self.load_data(root)
    
    def load_data(self, root):
        path_list = [path for path in os.listdir(root)]
        data = []
        label = []
        for data_path in path_list:
            data_org = np.load(os.path.join(root, data_path))
            if data_path[1] == '3':  # 1: hard fall 2: soft fall 3: non-fall
                lb = 0
            else:
                lb = 1
            data.append(data_org)
            label.append(lb)
        return np.array(data), np.array(label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_tensor = torch.from_numpy(self.data[idx]).float().unsqueeze(0)  # 添加通道维度
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor

# CNN模型定义
class LKCNN(nn.Module):
    def __init__(self):
        super(LKCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (7, 1), 1)
        self.pool1 = nn.MaxPool2d((4, 1), (4, 1))
        self.conv2 = nn.Conv2d(32, 64, (1, 9), 1)
        self.pool2 = nn.MaxPool2d((1, 4), (1, 4))
        self.fc1 = nn.Linear(64 * 30 * 14, 1024)  # 需确认这个维度是否正确
        self.fc2 = nn.Linear(1024, 2)  # 二分类

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train(model, device, train_loader, optimizer, epoch, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(model, device, test_loader, test_losses, test_accuracy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / len(test_loader.dataset))
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LKCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    name = torch.cuda.get_device_name()
    print('Using device '+ name + ' to train the model.')

    train_dataset = NPYDataset('/kaggle/working/diffusion')
    test_dataset = NPYDataset('/kaggle/input/fyp-dataset/12_DTweighted/test')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_losses, train_counter = [], []
    test_losses, test_accuracy = [], []

    for epoch in range(1, 11):  # 进行10个epochs的训练
        train(model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        test(model, device, test_loader, test_losses, test_accuracy)

    # 绘制loss和accuracy曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Training loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy (%)')

    plt.tight_layout()
    plt.show()
    plt.savefig("/kaggle/working/Classifier.jpg")

if __name__ == '__main__':
    main()
