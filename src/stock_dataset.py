import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, X: torch.tensor, y_reg: torch.tensor, y_class: torch.tensor):
        self.X = X
        self.y_reg = y_reg  # Regression target
        self.y_class = y_class  # Binary classification target

    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_class[idx]