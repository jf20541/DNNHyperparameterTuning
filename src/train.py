import pandas as pd
import numpy as np
import torch
import optuna
from dataset import HotelDataSet
from model import DeepNeuralNetwork
import config
import engine


def train(fold, params, save_model=False):
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
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    model = DeepNeuralNetwork(
        n_features=xtrain.shape[1],
        n_targets=ytrain.shape[1],
        n_layers=params["num_layers"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
    )


    optimizer = params['optimizer'](model.parameters(), lr=params["learning_rate"])

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

    return best_loss


def objective(trial):
    params = {
        'optimizer': trial.suggest_categorical('optimizer', [torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW]),
        "num_layers": trial.suggest_int("num_layers", 1, 7),
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
    }
    all_losses = []
    for f_ in range(1):
        temp_loss = train(f_, params, save_model=False)
        all_losses.append(temp_loss)

    return np.mean(all_losses)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    scores = 0
    for j in range(1):
        scr = train(j, trial_.params, save_model=True)
        scores += scr

    print(scores)
