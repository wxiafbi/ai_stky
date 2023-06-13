import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 从Excel文件中读取数据
def read_data_from_excel(file_name):
    data = pd.read_excel(file_name, sheet_name=None)
    return data

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# 定义模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 读取数据
file_name = '历史数据.xlsx'
sheets_data = read_data_from_excel(file_name)

# 处理数据
all_data = []
all_targets = []
for sheet_name, sheet_data in sheets_data.items():
    print(sheet_name)
    print(sheet_data)
    sheet_data = sheet_data.dropna()
    if sheet_data.empty:  # 检查sheet是否为空
        continue
    data = sheet_data.iloc[:, 1].values
    print(data)
    target = sheet_data.iloc[0, -1]
    all_data.append(data)
    all_targets.append(target)

# 数据标准化
scaler = StandardScaler()
all_data = [data.reshape(1, -1) for data in all_data]  # 将每个一维数组重塑为二维数组
all_data = scaler.fit_transform(all_data)
# 划分训练集和验证集
train_data, val_data, train_targets, val_targets = train_test_split(all_data, all_targets, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = CustomDataset(train_data, train_targets)
val_dataset = CustomDataset(val_data, val_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 超参数
input_size = train_data.shape[1]
hidden_size = 64
output_size = 1
num_epochs = 100
learning_rate = 0.001

# 创建模型实例
model = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# 验证模型
with torch.no_grad():
    val_loss = 0
    for data, targets in val_loader:
        data = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        outputs = model(data)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    print(f'Validation Loss: {val_loss / len(val_loader)}')

# 读取新的sheet并预测
new_sheet_name = 'new_sheet'
new_sheet_data = pd.read_excel(file_name, sheet_name=new_sheet_name)
new_sheet_data = new_sheet_data.dropna()
new_data = new_sheet_data.iloc[:, 1].values.reshape(1, -1)
new_data = scaler.transform(new_data)
new_data = torch.tensor(new_data, dtype=torch.float32)

with torch.no_grad():
    prediction = model(new_data)
    print(f'Prediction for new sheet: {prediction.item()}')
