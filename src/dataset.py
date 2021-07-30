import torch
from torch.utils.data import Dataset


class HotelDataSet(Dataset):
    # data loading
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    # return data length
    def __len__(self):
        return self.features.shape[0]

    # return item on the index
    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.features[idx], dtype=torch.float),
            "y": torch.tensor(self.targets[idx], dtype=torch.float),
        }
