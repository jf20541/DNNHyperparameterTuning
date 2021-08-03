import pandas as pd
import numpy as np
import torch
import optuna
from dataset import HotelDataSet
from model import DeepNeuralNetwork
import config
from engine2 import Engine2
from sklearn.metrics import roc_auc_score


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

    eng = Engine2(model, optimizer)

    best_metric = 0
    # early_stopping_iter = 10
    # early_stopping_counter = 0

    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)
        eval_outputs = np.array(eval_outputs) >= 0.5
        # calculating accuracy score
        train_metric = roc_auc_score(train_targets, train_outputs)
        eval_metric = roc_auc_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train ROC-AUC: {train_metric:.4f}, Eval ROC-AUC: {eval_metric:.4f}"
        )
            
        if eval_metric >= best_metric:
            best_metric = eval_metric
            if save_model:
                torch.save(model.state_dict(), f"../models/model{fold}.bin")
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter > early_stopping_iter:
        #     break
    return best_metric


if __name__ == "__main__":
    
    def objective(trial):
        params = {
            'optimizer': trial.suggest_categorical('optimizer', [torch.optim.Adam, torch.optim.AdamW]),
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 12, 84),
            "dropout": trial.suggest_uniform("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
        }
        all_metrics = []
        for i in range(1):
            temp_metric = train(i, params, save_model=False)
            all_metrics.append(temp_metric)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return np.mean(all_metrics)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    n_trial= study.best_trial
    print(f"Best Trial: {n_trial}, Value: {n_trial.values}")
    print(f'Best Parameters: {n_trial.params}')
    
    scores = 0
    for j in range(1):
        scr = train(j, n_trial.params, save_model=True)
        scores += scr

    print(f'SCORE: {scores}')
    
    
    # fig =optuna.visualization.plot_param_importances(study)
    # fig2 =optuna.visualization.plot_contour(study, params=['learning_rate', 'optimizer'])
    # fig.show()
    # fig2.show()