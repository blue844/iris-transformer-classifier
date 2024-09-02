import torch
import torch.nn as nn

# 定义与训练时相同的模型
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
        x = (x - self.mean) / self.scale
        x = x.unsqueeze(1)
        x = self.input_linear(x) + self.positional_encoding[:, :x.size(1)]

        # 使用相同的x作为目标序列
        tgt = x

        x = self.transformer(x, tgt)
        x = x.squeeze(1)
        x = self.output_linear(x)
        return x

# 加载模型
checkpoint = torch.load('my_transformer_with_params.pth')
model = MyTransformer(d_model=4, nhead=2, num_layers=2, dropout=0.1, mean=checkpoint['mean'], scale=checkpoint['scale'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入数据
new_data = [[5.6,2.9,3.6,1.3,]]  # 替换为你的新数据
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

# 进行预测
with torch.no_grad():
    outputs = model(new_data_tensor)
    predicted_class = torch.argmax(outputs, dim=1)
    print(f'Predicted class: {predicted_class.item()}')
