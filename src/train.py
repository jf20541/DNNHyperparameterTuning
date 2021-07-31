import config
from dataset import HotelDataSet
from torch.utils.data import DataLoader
import pandas as pd
from model import Model
import torch
import engine
import numpy as np

"""
Training Set: x_train (features), x_test (targets)
Testing Set: y_train (features), y_test (targets)
"""


def train(fold, save_model=False):
    df = pd.read_csv(config.TRAINING_FOLDS)

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    ytrain = train_df[["is_canceled"]].values
    xtrain = train_df.drop("is_canceled", axis=1).values

    yvalid = valid_df[["is_canceled"]].values
    xvalid = valid_df.drop("is_canceled", axis=1).values

    train_dataset = HotelDataSet(features=xtrain, targets=ytrain)
    test_dataset = HotelDataSet(features=xvalid, targets=yvalid)

    # initiate custom dataset and feed to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )

    model = Model(
        nfeatures=xtrain.shape[1],
        ntargets=ytrain.shape[1],
        nlayers=2,
        hidden_size=30,
        dropout=0.3,
    )
    print(model)
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


# def objective(trial):


if __name__ == "__main__":
    train(fold=0)
