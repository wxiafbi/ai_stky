import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 定义模型类
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(2, 1)  # 使用线性层作为模型

    def forward(self, x):
        out = self.fc(x)
        return out

# 读取Excel数据
def read_excel_data(file_path):
    data = pd.read_excel(file_path, sheet_name=None)  # 读取所有的sheet
    return data


# 处理数据并划分训练集和验证集
def process_data(data):
    all_data = []
    for sheet_name, sheet_data in data.items():
        if sheet_data.empty:
            continue  # 跳过空的sheet

        timestamps = pd.to_datetime(sheet_data.iloc[:, 0]).astype(np.int64)  # 将时间转换为时间戳
        values = sheet_data.iloc[:, 1].values

        if len(sheet_data.columns) > 2:
            target = sheet_data.iloc[:, 2].values[0]  # 获取要预测的目标值（第三列只有一个数值）
        else:
            target = None  # 如果没有目标值，则设为None

        # 归一化数据（可选）
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        values = (values - values.min()) / (values.max() - values.min())

        data_points = np.column_stack((timestamps, values))
        all_data.append((data_points, target))

    # 划分训练集和验证集
    train_data = all_data[:len(all_data) - 1]
    val_data = all_data[-1]

    return train_data, val_data

# 创建模型实例
model = Model()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 读取训练数据和验证数据
file_path = '历史数据.xlsx'
data = read_excel_data(file_path)
train_data, val_data = process_data(data)

if len(train_data) == 0 or val_data[1] is None:
    print("数据不足，无法训练模型。请检查Excel中的数据是否正确。")
    exit()

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()  # 设置模型为训练模式

    for input_data, target in train_data:
        inputs = torch.Tensor(input_data)
        target = torch.Tensor([target])

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, target)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_data)

    # 在验证集上计算损失
    val_inputs, val_target = val_data
    val_inputs = torch.Tensor(val_inputs)

    model.eval()  # 设置模型为评估模式
    val_outputs = model(val_inputs)

    if val_target is not None:
        val_target = torch.Tensor([val_target])
        val_loss = criterion(val_outputs, val_target)
        print(f'Epoch [{epoch+1}/{num_epochs}],"||", Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}],'|' ,Train Loss: {avg_train_loss:.4f}, Val Loss: N/A')


# 读取新的Excel数据并进行预测
new_data = read_excel_data('验证数据.xlsx')
for sheet_name, sheet_data in new_data.items():
    timestamps = pd.to_datetime(sheet_data.iloc[:, 0]).astype(np.int64)  # 将时间转换为时间戳
    values = sheet_data.iloc[:, 1].values

    # 归一化数据（可选）
    timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    values = (values - values.min()) / (values.max() - values.min())

    new_inputs = torch.Tensor(np.column_stack((timestamps, values)))
    predicted_output = model(new_inputs)

    print(f'Predicted values for sheet {sheet_name}: {predicted_output.item()}')
