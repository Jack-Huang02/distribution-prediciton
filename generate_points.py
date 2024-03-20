import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def init_weight(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') == 1:
        m.weight.data.normal_(0.0, 0.02)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.apply(init_weight)

    def forward(self, x):
        return self.fc(x)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.apply(init_weight)
    def forward(self, x):
        return self.fc(x)

df = pd.read_csv('data/total_mouse.csv')
# 筛选目标大小为 32 的数据行
filtered_data = df[(df['TargetWidth'] == 32) & (df['Velocity'] == 64)]
input_data = filtered_data[['X', 'Y']].values.astype(np.float32)
scaler = StandardScaler()
input_data_normalized = scaler.fit_transform(input_data)
data = torch.tensor(input_data_normalized, dtype=torch.float32)
print(len(data))
# 设置参数
input_dim = 2  # 输入特征的维度，即 X 和 Y 两列数据
output_dim = 2  # 生成器输出的维度，根据你的任务需求设置
learning_rate = 0.0002
batch_size = 32
epochs = 5000
# patience = 10
# current_patience = patience
# best_loss = np.inf

# 初始化生成器模型、损失函数和优化器
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

criterion = nn.BCELoss()  # 使用均方误差损失
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

real_data_for_visualization = data.numpy()

# 训练生成器模型
for epoch in range(epochs):
    for i in range(len(data) // batch_size):
        real_data = data[i * batch_size: (i + 1) * batch_size]
        real_labels = torch.ones(batch_size, 1)
        noise = torch.randn(batch_size, input_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)

        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data)
        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
    noise = torch.randn(batch_size, input_dim)
    fake_data = generator(noise)
    fake_labels = torch.ones(batch_size, 1)

    output = discriminator(fake_data)
    g_loss = criterion(output, fake_labels)

    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
# 验证模型并检查是否提前结束训练
#     if epoch % 500 == 0:
#         with torch.no_grad():
#             noise = torch.randn(batch_size, input_dim)
#             fake_data = generator(noise)
#             fake_labels = torch.ones(batch_size, 1)
#             output = discriminator(fake_data)
#             valid_loss = criterion(output, fake_labels).item()
#             if valid_loss < best_loss:
#                 best_loss = valid_loss
#                 current_patience = patience
#             else:
#                 current_patience -= 1
#                 if current_patience == 0:
#                     print("Early stopping!")
#                     break



# 生成坐标数据
def generate_coordinates(num_samples):
    noise = torch.randn(num_samples, input_dim)
    generated_data = generator(noise).detach().numpy()
    generated_data = scaler.inverse_transform(generated_data)  # 反标准化
    return generated_data


# 生成坐标数据并可视化
generated_coordinates = generate_coordinates(200)

# 原始数据
original_mean = np.mean(input_data, axis=0)
original_std = np.std(input_data, axis=0)

# 生成数据
generated_mean = np.mean(generated_coordinates, axis=0)
generated_std = np.std(generated_coordinates, axis=0)

print("原始数据均值：", original_mean)
print("原始数据方差：", original_std)
print("生成数据均值：", generated_mean)
print("生成数据方差：", generated_std)

plt.scatter(generated_coordinates[:, 0], generated_coordinates[:, 1], label='Generated Data')
plt.scatter(input_data[:, 0], input_data[:, 1], label='Real Data')
plt.legend()
plt.show()

# # 可视化生成的数据和真实数据
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(real_data_for_visualization[:, 0], real_data_for_visualization[:, 1], label='Real Data', alpha=0.5)
# plt.title('Real Data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data', alpha=0.5)
# plt.title('Generated Data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# plt.show()