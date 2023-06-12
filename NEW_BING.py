import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 读取Excel文件中的数据
def read_data_from_excel(file_path):
    data = {}
    xls = pd.read_excel(file_path, sheet_name=None)
    for sheet_name in xls:
        data[sheet_name] = xls[sheet_name]
    return data

# 定义数据集
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx, 1:3].values
        if self.transform:
            sample = self.transform(sample)

        return sample

# 定义模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# 训练模型
def train_model(model, train_data, criterion, optimizer, num_epochs=1000):
    for epoch in range(num_epochs):
        for i, data in enumerate(train_data):
            inputs = data[:, 0:2]
            labels = data[:, 2]

            # 前向传播
            y_pred = model(inputs)

            # 计算损失
            loss = criterion(y_pred, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 验证模型
def validate_model(model, validation_data):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_data:
            inputs = data[:, 0:2]
            labels = data[:, 2]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation data: %d %%' % (
        100 * correct / total))

# 预测新数据
def predict(model, new_data):
    with torch.no_grad():
        inputs = new_data[:, 0:2]
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

# 主程序
if __name__ == '__main__':
    # 读取数据
    file_path = 'data.xlsx'
    data_dict = read_data_from_excel(file_path)

    # 划分训练数据和验证数据
    train_data_list = []
    validation_data_list = []
    for sheet_name in data_dict:
        sheet_data = data_dict[sheet_name]
        train_size = int(0.8 * len(sheet_data))
        train_data_list.append(sheet_data.iloc[:train_size])
        validation_data_list.append(sheet_data.iloc[train_size:])

    train_data_df = pd.concat(train_data_list)
    validation_data_df = pd.concat(validation_data_list)

    train_dataset = MyDataset(train_data_df)
    validation_dataset = MyDataset(validation_data_df)

    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4,
                                       shuffle=False, num_workers=0)

    # 定义模型、损失函数和优化器
    model = MyModel()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # 训练模型
    train_model(model, train_dataloader, criterion, optimizer)

    # 验证模型
    validate_model(model, validation_dataloader)

    # 预测新数据
    new_sheet_name = 'new_sheet'
    new_sheet_data_df = pd.read_excel(file_path, sheet_name=new_sheet_name)
