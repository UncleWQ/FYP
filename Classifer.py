import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from sklearn.metrics import recall_score, precision_score

class NPYDataset(Dataset):
    def __init__(self, root):
        self.data, self.labels = self.load_data(root)
    
    def load_data(self, root):
        path_list = [path for path in os.listdir(root)]
        data = []
        label = []
        for data_path in path_list:
            data_org = np.load(os.path.join(root, data_path))
            if data_path[1] == '3':  # 非跌倒
                lb = 0
            else:  # 跌倒
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

class LKCNN(nn.Module):
    def __init__(self):
        super(LKCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (7, 1), 1)
        self.pool1 = nn.MaxPool2d((4, 1), (4, 1))
        self.conv2 = nn.Conv2d(32, 64, (1, 9), 1)
        self.pool2 = nn.MaxPool2d((1, 4), (1, 4))
        self.fc1 = nn.Linear(64 * 30 * 14, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch, epoch_losses):
    model.train()
    train_loss = 0
    total_samples = len(train_loader.dataset)  # 获取训练集的总样本数

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)  # 累加每个批次的损失，乘以批次内的样本数

    average_loss = train_loss / total_samples  # 使用总样本数计算平均损失
    average_loss = average_loss / 100
    epoch_losses.append(average_loss)
    print(f'Train Epoch: {epoch}\tAverage Loss: {average_loss:.6f}')

def test(model, device, test_loader, test_losses, test_accuracy, test_recall, test_precision):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_outputs.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / len(test_loader.dataset))
    recall = 100 * recall_score(all_targets, all_outputs)  # Multiply by 100 for percentage
    precision = 100 * precision_score(all_targets, all_outputs)  # Multiply by 100 for percentage
    test_recall.append(recall)
    test_precision.append(precision)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%), Recall: {recall:.2f}%, Precision: {precision:.2f}%\n')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LKCNN().to(device)
    name = torch.cuda.get_device_name()
    print('Using device '+ name + ' to train the model.')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_dataset = NPYDataset('/tmp/FYP_Projects/Diffusion')
    test_dataset = NPYDataset('/tmp/FYP_Projects/12_DTweighted/test')
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    print("Data loaded!")

    epoch_losses = []
    test_losses, test_accuracy, test_recall, test_precision = [], [], [], []

    begin = time.time()
    for epoch in range(1, 201):
        train(model, device, train_loader, optimizer, epoch, epoch_losses)
        test(model, device, test_loader, test_losses, test_accuracy, test_recall, test_precision)


    # 计算所有周期的平均损失、准确率、召回率和精确率
    average_training_loss = sum(epoch_losses) / len(epoch_losses)
    average_test_accuracy = sum(test_accuracy) / len(test_accuracy)
    average_test_recall = sum(test_recall) / len(test_recall)
    average_test_precision = sum(test_precision) / len(test_precision)

    # 打印计算得到的平均值
    print(f'Average Training Loss over {len(epoch_losses)} epochs: {average_training_loss:.6f}')
    print(f'Average Test Accuracy over {len(test_accuracy)} epochs: {average_test_accuracy:.3f}%')
    print(f'Average Test Recall over {len(test_recall)} epochs: {average_test_recall:.3f}%')
    print(f'Average Test Precision over {len(test_precision)} epochs: {average_test_precision:.3f}%')

    plt.figure(figsize=(24, 20))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(test_accuracy)+1), test_accuracy, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(test_recall)+1), test_recall, color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Test Recall (%)')

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(test_precision)+1), test_precision, color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Test Precision (%)')

    plt.tight_layout()
    plt.show()
    plt.savefig("Classifier_performance_diffusion.jpg")


    elapsed = (time.time() - begin) / 60
    print(f'Task finished! Total time={elapsed:.3f} min')

if __name__ == '__main__':
    main()
