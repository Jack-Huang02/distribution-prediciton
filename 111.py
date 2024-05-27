import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dataset import getdata
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出两个坐标 x 和 y
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, coords):
        return self.model(coords)

# 假设 coords 是一个形状为 (200, 2) 的 numpy 数组
X, Y = getdata(32, 64)
coords = np.array(list(zip(X, Y)))  # 模拟数据，应替换为实际数据
iso = IsolationForest(contamination=0.02)
outliers = iso.fit_predict(coords)
clean_coords = coords[outliers != -1]

scaler = StandardScaler()
input_data_normalized = scaler.fit_transform(clean_coords)
data = torch.tensor(input_data_normalized, dtype=torch.float32)

# 创建 DataLoader
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
generator = Generator()
discriminator = Discriminator()

# 优化器
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4)
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4)

# 损失函数
loss_function = nn.BCELoss()

d_loss_list = []
g_loss_list = []
# 训练循环
epochs = 100
for epoch in range(epochs):
    for real_coords, in dataloader:

        # 训练判别器
        discriminator.zero_grad()
        real_preds = discriminator(real_coords)
        real_targets = torch.ones(real_coords.size(0), 1)
        real_loss = loss_function(real_preds, real_targets)
        
        z = torch.randn(real_coords.size(0), 100)
        fake_coords = generator(z)
        fake_preds = discriminator(fake_coords.detach())
        fake_targets = torch.zeros(real_coords.size(0), 1)
        fake_loss = loss_function(fake_preds, fake_targets)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # 训练生成器
        generator.zero_grad()
        trick_preds = discriminator(fake_coords)
        trick_targets = torch.ones(real_coords.size(0), 1)
        g_loss = loss_function(trick_preds, trick_targets)
        g_loss.backward()
        g_optimizer.step()
    
    d_loss_list.append(d_loss.item())
    g_loss_list.append(g_loss.item())

    print(f'Epoch {epoch+1}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}')
# 保存模型参数目
torch.save(generator.state_dict(), './model/generator.pth')
torch.save(discriminator.state_dict(), './model/discriminator.pth')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(d_loss_list, label='Discriminator Loss', color='blue')
plt.plot(g_loss_list, label='Generator Loss', color='red')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 生成随机噪声
z = torch.randn(130, 100)  # 生成1000个样本，每个样本是100维的随机噪声

# 使用生成器生成落点坐标
with torch.no_grad():  # 关闭梯度计算
    generated_coords = generator(z).numpy()  # 转换为NumPy数组
original_data_scaled = scaler.inverse_transform(generated_coords)



plt.scatter(clean_coords[:, 0], clean_coords[:, 1], color='blue', alpha=0.5, label='Original Data')

# 可视化生成的反归一化数据
plt.scatter(original_data_scaled[:, 0], original_data_scaled[:, 1], color='red', alpha=0.5, label='Generated Data')

plt.title('Comparison of Original and Generated Data')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()