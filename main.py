import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import gaussian_kde

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 从Excel读取数据
excel_file = r"C:\Users\erasme153\Desktop\chemical.xlsx"
df = pd.read_excel(excel_file, header=None, skiprows=1)

# 提取第三列的输出数据
output_data = df.iloc[:, 2].values

# 提取第四列到第11列的输入参数数据
input_data = df.iloc[:, 3:11].values

# 数据标准化
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# 数据预处理和划分训练集和测试集
input_train, input_test, output_train, output_test = train_test_split(
    input_data, output_data, test_size=0.2
)

# 超参数
input_dim = input_data.shape[1]
output_dim = 1
latent_dim = 10
num_epochs = 1000
learning_rate = 0.0005

# FreeBits参数
threshold = 0.01  # 阈值，用于确定自由位
tau = 10.0  # 初始tau值

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoding = self.encoder(x)
        mu, logvar = encoding[:, :self.latent_dim], encoding[:, self.latent_dim:]

        # 对 z 抽样十次，每一次将结果送入decoder得到一个 x_hat ，取 x_hat 的均值返回
        x_hat = 0
        for i in range(10):
            z = self.reparameterize(mu, logvar)
            x_hat += self.decoder(z)
        x_hat = x_hat/10

        return x_hat, mu, logvar

# 创建 VAE 模型
vae = VAE(input_dim, latent_dim)

# 初始化优化器
optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-4)

# 初始化损失列表
recon_loss_list = []
kl_divergence_list = []

# 训练 VAE 模型
vae.train()
for epoch in range(num_epochs):
    input_train_tensor = torch.tensor(input_train, dtype=torch.float32)
    output_train_tensor = torch.tensor(output_train, dtype=torch.float32).view(-1, 1)
    x_hat, mu, logvar = vae(input_train_tensor)
    recon_loss = nn.MSELoss()(x_hat, output_train_tensor)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 使用 FreeBits 逐渐增加 KL 散度的权重
    free_bits = torch.max(threshold - kl_divergence, torch.tensor(0.0))
    loss = recon_loss + tau * free_bits

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 保存损失值到列表中
    recon_loss_list.append(recon_loss.item())
    kl_divergence_list.append(kl_divergence.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, FreeBits: {free_bits.item():.4f}')

    # 根据需要更新tau
    if epoch < num_epochs / 2:
        tau = 10.0  # 在前半程训练过程中保持初始值
    else:
        tau = 0.01  # 在后半程训练过程中减小tau，使KL散度更重要

# 将损失函数转换为 log
log_recon_loss_list = np.log(recon_loss_list)
log_kl_divergence_list = np.log(kl_divergence_list)

# 使用模型生成输出
vae.eval()
with torch.no_grad():
    input_test_tensor = torch.tensor(input_test, dtype=torch.float32)
    generated_output, _, _ = vae(input_test_tensor)

# 计算生成数据和原数据的 MAE
mae_value = mean_absolute_error(output_test, generated_output.numpy())
print(f'MAE Score: {mae_value:.4f}')

# 绘制原始数据和生成数据的核密度估计曲线以及损失曲线
plt.figure(figsize=(12, 6), dpi=300)

plt.subplot(2, 1, 1)
# 绘制原始数据的 KDE 曲线
kde_original = gaussian_kde(output_data)
x_original = np.linspace(min(output_data), max(output_data), 1000)
y_original = kde_original(x_original)
plt.plot(x_original, y_original, label='原始完成液流量数据', color='blue')

# 绘制生成数据的 KDE 曲线
kde_generated = gaussian_kde(generated_output.flatten())
x_generated = np.linspace(min(generated_output.flatten()), max(generated_output.flatten()), 1000)
y_generated = kde_generated(x_generated)
plt.plot(x_generated, y_generated, label='生成完成液流量数据', color='red')

plt.xlabel('完成液流量值')
plt.ylabel('完成液流量密度')
plt.title('原始完成液流量 VS 生成完成液流量')

plt.figure(figsize=(12, 6), dpi=300)  # 设置图像分辨率为300dpi
plt.subplot(2, 1, 2)  # 新增一个子图用于绘制损失曲线

# 绘制重建损失曲线
plt.plot(range(num_epochs), log_recon_loss_list, label='Log Reconstruction Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Log Reconstruction Loss')

# 绘制KL散度曲线
plt.plot(range(num_epochs), log_kl_divergence_list, label='Log KL Divergence', color='red')
plt.xlabel('Epoch')
plt.ylabel('Log KL Divergence')

plt.title('Loss Curves')
plt.legend()

# 保存图像到本地
save_path = r"C:\Users\erasme153\Pictures\data_distribution.png"
plt.savefig(save_path)

plt.show()
