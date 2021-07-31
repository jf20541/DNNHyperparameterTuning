import torch
from torch.utils.data import Dataset
import pandas as pd
import config


class HotelDataSet:
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
            "x": torch.tensor(self.features[idx, :], dtype=torch.float),
            "y": torch.tensor(self.targets[idx], dtype=torch.float),
        }


# def train(fold):
#     df = pd.read_csv(config.TRAINING_FOLDS)

#     train_df = df[df.kfold != fold].reset_index(drop=True)
#     valid_df = df[df.kfold == fold].reset_index(drop=True)

#     ytrain = train_df.is_canceled.values
#     xtrain = train_df.drop('is_canceled', axis=1).values

#     yvalid = valid_df.is_canceled.values
#     xvalid = valid_df.drop('is_canceled', axis=1).values

#     # print(ytrain.shape, xtrain.shape)     # (82438,) (82438, 31)
#     # print(yvalid.shape, xvalid.shape)     # (20610,) (20610, 31)

#     train_dataset = HotelDataSet(features=xtrain, targets=ytrain)
#     test_dataset = HotelDataSet(features=xvalid, targets=yvalid)

#     print(len(train_dataset))
#     print(len(test_dataset))
#     print(train_dataset[0])

#     train_loder = torch.utils.data.DataLoader(train_dataset, batch_size=256)
# for idx in train_loader:
#     print(idx['x'].shape)
#     print(idx['y'].shape)
#     break

# for idx in test_loader:
#     print(idx['x'].shape)
#     print(idx['y'].shape)
#     break

# train(fold=0)
