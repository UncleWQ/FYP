# FYP

## 一、实验流程

对于给定的数据集，训练集85000，测试集11200

### 数据类型：

（128，64）二维数组，单通道，8192个数据点，保存为.npy文件

### 数据文件命名:

1.data_path[1] 代表hard, soft, non-fall

2.data_path[2] 和 data_path[3]两位代表28个属性

3.数据集里面Hard:1-9, soft:10-14, non-fall:15-28

### Baseline：

#### (1)对于此数据集，看成（128，64，1）图像

利用diffusion models对该数据集进行数据增强

每个数据至少生成3个变体

#### (2)将生成的变体保存到新文件夹train_diffusion_augmentation中

利用学长给的代码，上标签：

```Py
import numpy as np
import os

def load_data(root):
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
    return np.array(data),np.array(label)

if __name__=='__main__':
    os.chdir('../12_DTweighted')

    ROOT = '../12_DTweighted/'
    train_path = ROOT + 'train/'
    test_path = ROOT + 'test/'
    trainX,trainy = load_data(train_path)
    testX,testy = load_data(test_path)

    print(load_data(train_path))
    print(load_data(test_path))
    
    print("data loaded")
```

注：变体和原数据的label是一样的

#### (3)搭卷积神经网络二分类

构建卷积神经网络实现二分类

![image-20240411194401110](C:\Users\王钦\AppData\Roaming\Typora\typora-user-images\image-20240411194401110.png)

把之前的标签01二分类

#### (4)测试结果

将原始数据集train、增强后的数据集train_diffusin_augmentation、train_augmentation_GAN

导入到上述的二分类模型，比较效果



## 二、扩散模型（Stable Diffusion)

![image-20240412222343573](C:\Users\王钦\AppData\Roaming\Typora\typora-user-images\image-20240412222343573.png)

```python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

# 检查CUDA是否可用，选择正确的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用CNN的扩散模型架构
class CNNDiffusionModel(nn.Module):
    def __init__(self, channels, time_embedding_size=32):
        super(CNNDiffusionModel, self).__init__()
        self.time_embedding = nn.Linear(1, time_embedding_size)

        # 增加了批量归一化层和池化层
        # 卷积层
        self.conv1 = nn.Conv2d(channels + time_embedding_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 池化层
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, channels, kernel_size=3, padding=1)

        # 全连接层
        self.fc = nn.Linear(channels * 32 * 16, channels * 128 * 64)  # Adjust according to the output size after pooling
        self.relu = nn.ReLU()

    def forward(self, x, t):
        t = self.time_embedding(t)  # 时间嵌入
        t = t.unsqueeze(-1).unsqueeze(-1)  # 改变t的形状以匹配x
        t = t.expand(-1, -1, x.shape[2], x.shape[3])

        x = torch.cat([x, t], dim=1)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 1, 128, 64)
        return x

# 扩散和逆扩散步骤
def diffusion_step(x, beta=0.1):
    noise = torch.randn_like(x).to(device)
    return np.sqrt(1 - beta) * x + np.sqrt(beta) * noise

def reverse_diffusion_step(x_t, t, model):
    pred_noise = model(x_t, t)
    beta = 0.1
    return (x_t - np.sqrt(beta) * pred_noise) / np.sqrt(1 - beta)

# 训练循环
def train(model, data_loader, epochs=1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for batch in data_loader:  # 这里直接迭代整个批次
            x = batch.to(device)  # 将批次移动到设备上
            t = torch.rand(x.size(0), 1).to(device)  # 生成与批次大小相同的随机时间步
            x_t = diffusion_step(x, beta=0.1)  # 执行扩散步骤
            pred_noise = model(x_t, t)  # 使用模型预测噪声
            true_noise = x - diffusion_step(x, beta=0.1)  # 计算真实噪声
            loss = loss_fn(pred_noise, true_noise)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# 采样过程
def sample(model, num_samples, img_height, img_width, channels):
    model.eval()
    x_t = torch.randn(num_samples, channels, img_height, img_width).to(device)
    timesteps = 1000

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            time_step = torch.full((num_samples, 1), fill_value=t/timesteps, device=device)
            x_t = reverse_diffusion_step(x_t, time_step, model)
    
    return x_t



class Dataset_file(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # 通道
        return data, file_path
    
class NumpyDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.file_names = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        data = np.load(file_path)  # 加载数据
        data = data.astype(np.float32)  # 确保数据类型为float32
        data = torch.from_numpy(data).unsqueeze(0)  # 转换为torch.Tensor并添加通道维度
        return data

def create_data_loader(directory, batch_size=32, num_workers=4):
    dataset = NumpyDataset(directory)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,       # 为了训练的随机性，打乱数据
        num_workers=num_workers  # 根据你的系统配置设置适当的工作线程数
    )
    return loader

# 初始化设备和模型


if __name__ == "__main__":
    model = CNNDiffusionModel(channels=1).to(device)
    model.train()
    train_loader = create_data_loader('/kaggle/input/fyp-dataset/12_DTweighted/train', batch_size=32, num_workers=8)
    train(model, train_loader, epochs=10)

    model.eval()  # 设置为评估模式

    # 创建数据集和数据加载器
    dataset = Dataset_file(root_dir='/kaggle/input/fyp-dataset/12_DTweighted/train')
    file_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 创建新目录
    new_dir = '/kaggle/working/train_diffusion_augmentation'
    os.makedirs(new_dir, exist_ok=True)

    # 采样并保存数据
    with torch.no_grad():
        for _, paths in file_loader:
            for path in paths:
                filename = os.path.basename(path)
                name, ext = os.path.splitext(filename)
                # 对每个数据生成三个变体
                generated_samples = sample(model, 3, 128, 64, 1)
                for j in range(1, 4):
                    new_filename = f"{name}-{j:02}{ext}"
                    np.save(os.path.join(new_dir, new_filename), generated_samples[j-1].cpu().numpy().squeeze(0))  # 保存生成的数据

    print("Task finished! Successfully Saved!")

```

