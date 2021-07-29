import config
from dataset import HotelDataSet
from torch.utils.data import DataLoader
import pandas as pd
from model import DeepNeuralNetwork
import torch


def train(fold):
    df = pd.read_csv(config.TRAINING_FOLDS)

    # training and testing data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # drop the target column and convert to numpy array
    x_train = df_train.drop("is_canceled", axis=1).values
    y_train = df_train.is_canceled.values

    x_test = df_test.drop("is_canceled", axis=1).values
    y_test = df_test.is_canceled.values

    # initiate custom dataset and feed to dataloader
    train_dataset = HotelDataSet(x_train, y_train)
    test_dataset = HotelDataSet(y_train, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE)

    model = DeepNeuralNetwork(
        n_features=x_train.shape[1], 
        n_targets=y_train.shape[1], 
        n_layers=2, 
        hidden_size=30, 
        dropout=0.3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    train(1)
