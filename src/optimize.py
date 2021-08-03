import torch
import numpy as np
import optuna

def objective(trial, train):
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
