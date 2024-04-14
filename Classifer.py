import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

class NPYDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.file_paths = [os.path.join(root, fname) for fname in os.listdir(root)]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        label = 0 if file_path.split('/')[-1][1] == '3' else 1
        data = torch.from_numpy(data).float().unsqueeze(0)  # 添加通道维度
        return data, label

class LKCNN(nn.Module):
    def __init__(self):
        super(LKCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (7, 1), 1)
        self.pool1 = nn.MaxPool2d((4, 1), (4, 1))
        self.conv2 = nn.Conv2d(32, 64, (1, 9), 1)
        self.pool2 = nn.MaxPool2d((1, 4), (1, 4))
        self.fc1 = nn.Linear(64 * 30 * 14, 1024)  # 确认这个维度是否正确
        self.fc2 = nn.Linear(1024, 2)  # 二分类

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LKCNN().to(device)
    name = torch.cuda.get_device_name()
    print('Using device '+ name + ' to train the model.')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = NPYDataset('/tmp/FYP_Projects/12_DTweighted/train')
    test_dataset = NPYDataset('/tmp/FYP_Projects/12_DTweighted/test')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Data loaded!")
    train_losses, train_counter = [], []
    test_losses, test_accuracy = [], []

    for epoch in range(1, 11):  # 进行10个epochs的训练
        train(model, device, train_loader, optimizer, epoch, train_losses, train_counter)
        test(model, device, test_loader, test_losses, test_accuracy)

    # 绘制loss和accuracy曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(train_counter, train_losses, color='blue')
    ax1.set_xlabel('Number of training examples seen')
    ax1.set_ylabel('Training loss')

    ax2.plot(test_accuracy, color='red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Test accuracy (%)')

    plt.tight_layout()
    plt.savefig("/tmp/FYP_Projects/Classifier.jpg")

if __name__ == '__main__':
    main()


