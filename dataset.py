import torch
from torch.utils.data import Dataset
import scipy.io as scio
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # 每个样本是一个二维数组，形状为 (1387, 2)
        feature = torch.tensor(sample[:, 0], dtype=torch.float32).unsqueeze(-1)  # 第一列是真实数据
        label = torch.tensor(sample[:, 1], dtype=torch.float32).unsqueeze(-1)  # 第二列是期望的数据
        return feature, label


def load_data(train_path, val_path, test_path):
    train_data_ = scio.loadmat(train_path)
    train_data_cells = train_data_['data'].squeeze()  # 获取所有cell
    val_data_ = scio.loadmat(val_path)
    val_data_cells = val_data_['data'].squeeze()  # 获取所有cell
    test_data_ = scio.loadmat(test_path)
    test_data_cells = test_data_['data'].squeeze()  # 获取所有cell

    train_dataset = CustomDataset(train_data_cells)
    val_dataset = CustomDataset(val_data_cells)
    test_dataset = CustomDataset(test_data_cells)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader
