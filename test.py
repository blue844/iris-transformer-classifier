import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 定义Transformer模型
class MyTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, mean, scale):
        super(MyTransformer, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=False)
        self.input_linear = nn.Linear(d_model, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 3)  # Assuming 3 classes for classification

    def forward(self, x):
        # 标准化
        x = (x - self.mean) / self.scale
        x = x.unsqueeze(1)
        x = self.input_linear(x) + self.positional_encoding[:, :x.size(1)]

        # 使用相同的x作为目标序列（这是一个简化示例）
        tgt = x

        x = self.transformer(x, tgt)
        x = x.squeeze(1)
        x = self.output_linear(x)
        return x

# 生成示例数据
iris = load_iris()
X = iris.data
y = iris.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算均值和标准差
scaler = StandardScaler()
scaler.fit(X_train)
mean, scale = scaler.mean_, scaler.scale_

# 保存均值和标准差
mean, scale = mean.tolist(), scale.tolist()

# 定义模型
d_model = X_train.shape[1]
nhead = 2
num_layers = 2
dropout = 0.1

model = MyTransformer(d_model, nhead, num_layers, dropout, mean, scale)

# 训练模型
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 转换数据为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# 训练循环
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': mean,
    'scale': scale,
}, 'my_transformer_with_params.pth')

print("Model trained and saved successfully.")
