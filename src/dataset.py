import torch


class HotelDataSet:
    # data loading
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    # return data length
    def __len__(self):
        return self.features.shape[0]

    # return item as tensors
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx, :], dtype=torch.float),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }
