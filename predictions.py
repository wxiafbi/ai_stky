import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x



def read_data_from_excel(file_path):
    data_frames = []
    excel_file = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, sheet_data in excel_file.items():
        sheet_data['datetime'] = pd.to_datetime(sheet_data.iloc[:, 0])
        sheet_data['timestamp'] = (sheet_data['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        sheet_data['float'] = sheet_data.iloc[:, 1]
        target = sheet_data.iloc[0, 2]
        data_frames.append((sheet_data[['timestamp', 'float']], target))
    return data_frames

def train_model(model, data_frames, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data, target in data_frames:
            inputs = torch.tensor(data.values, dtype=torch.float32)
            labels = torch.tensor(target, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def predict(model, data_frame):
    inputs = torch.tensor(data_frame.values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy()

file_path = '历史数据.xlsx'
data_frames = read_data_from_excel(file_path)

model = SimpleModel()
train_model(model, data_frames)

new_sheet_data = read_data_from_excel('验证数据.xlsx')[0][0]
predictions = predict(model, new_sheet_data)
print(predictions)
