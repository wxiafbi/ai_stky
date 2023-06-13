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


# 读取数据
file_name = '历史数据.xlsx'
sheets_data = read_data_from_excel(file_name)
# print(sheets_data.items())

# 处理数据
all_data = []
all_targets = []
for sheet_name, sheet_data in sheets_data.items():
    # print(sheet_name)
    print(sheet_data)
    sheet_data = sheet_data.dropna()
    data = sheet_data.iloc[:, 1].values
    print(data)
    print(type(data))

    target = sheet_data.iloc[0, 2]
    all_data.append(data)
    all_targets.append(target)
