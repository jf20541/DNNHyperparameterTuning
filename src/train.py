import config
from dataset import HotelDataSet
from torch.utils.data import DataLoader
import pandas as pd
from model import DeepNeuralNetwork
import torch
import engine
import numpy as np


def train(fold, save_model=False):
    df = pd.read_csv(config.TRAINING_FOLDS)

    features_col = df.drop("is_canceled", axis=1).columns

    # training and testing data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # drop the target column and convert to numpy array
    x_train = df_train[features_col].values
    y_train = df_train[features_col].values

    x_test = df_train["is_canceled"].values
    y_test = df_test["is_canceled"].values

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # initiate custom dataset and feed to dataloader
    train_dataset = HotelDataSet(x_train, y_train)
    test_dataset = HotelDataSet(x_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )

    model = DeepNeuralNetwork(
        n_features=x_train.shape[1],
        n_targets=y_train.shape[1],
        n_layers=2,
        hidden_size=30,
        dropout=0.3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    eng = engine.Engine(model, optimizer)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.evaluate(test_loader)
        print(
            f"Fold:{fold}, Epoch:{epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}"
        )
        if test_loss <= best_loss:
            best_loss = test_loss
            if save_model:
                torch.save(model.state_dict(), f"../models/model{fold}.bin")
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break


if __name__ == "__main__":
    train(fold=0)
