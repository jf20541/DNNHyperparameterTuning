import pandas as pd
import numpy as np
import torch
import optuna
from dataset import HotelDataSet
from model import DeepNeuralNetwork
import config
from engine import Engine
from sklearn.metrics import roc_auc_score
import torch.optim as optim


def train(fold, params, save_model=False):
    """[summary]
    Args:
        fold ([int]): [Stratified 5-Fold (avoids overfitting) ]
        params ([dict]): [define a combination of hyperparameters]
        save_model (bool, optional): [save optimal model's parameters]. Defaults to False.
    Returns:
        [float]: [optimal ROC-AUC metric]
    """
    df = pd.read_csv(config.TRAINING_FOLDS)

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    # split the data into training and testing set (define features, target) values
    y_train = train_df[["is_canceled"]].values
    x_train = train_df.drop("is_canceled", axis=1).values

    y_test = valid_df[["is_canceled"]].values
    x_test = valid_df.drop("is_canceled", axis=1).values

    # feed the data into custom Dataset
    train_dataset = HotelDataSet(x_train, y_train)
    test_dataset = HotelDataSet(x_test, y_test)

    # initiate custom dataset and feed to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )
    # inititate DNN with params
    model = DeepNeuralNetwork(
        n_features=x_train.shape[1],
        n_targets=y_train.shape[1],
        n_layers=params["num_layers"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
    )

    optimizer = params["optimizer"](model.parameters(), lr=params["learning_rate"])
    eng = Engine(model, optimizer)

    best_metric = 0
    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)
        train_outputs = np.array(eval_outputs) >= 0.5
        eval_outputs = np.array(eval_outputs) >= 0.5
        # calculating roc-auc score for train&eval
        train_metric = roc_auc_score(train_targets, train_outputs)
        eval_metric = roc_auc_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train ROC-AUC: {train_metric:.4f}, Eval ROC-AUC: {eval_metric:.4f}"
        )
        # save optimal metrics to model.bin
        if eval_metric > best_metric:
            best_metric = eval_metric
            if save_model:
                torch.save(model.state_dict(), f"../models/model{fold}.bin")

    return best_metric


if __name__ == "__main__":

    def objective(trial):
        """[define a combination of hyperparameters]
        Args:
            trial ([type]): [trial object is used to construct a model inside the objective function]
        Raises:
            optuna.exceptions.TrialPruned: [If pruned, we go to next n_trials]
        Returns:
            [type]: [the value that Optuna will optimize]
        """
        params = {
            "optimizer": trial.suggest_categorical(
                "optimizer", [optim.SGD, optim.Adam, optim.AdamW]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 10),
            "hidden_size": trial.suggest_int("hidden_size", 2, 112),
            "dropout": trial.suggest_uniform("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.01),
        }
        all_metrics = []
        for i in range(1):
            temp_metric = train(i, params, save_model=False)
            all_metrics.append(temp_metric)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return np.mean(all_metrics)

    # study object contains information about the required parameter space
    # increase the return value of our optimization function
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(), direction="maximize"
    )
    # initiate optimize with 10 trials
    study.optimize(objective, n_trials=10)

    # define number of pruned&completed trials (saves time and computing power)
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # print metric and optimal combiniation of hyperparameters
    n_trial = study.best_trial
    print(f"Best Trial: {n_trial}, Value: {n_trial.values}")
    print(f"Best Parameters: {n_trial.params}")

    scores = 0
    for j in range(1):
        scr = train(j, n_trial.params, save_model=True)
        scores += scr
    # plot param importance and contour
    fig = optuna.visualization.plot_param_importances(study)
    fig2 = optuna.visualization.plot_contour(
        study, params=["learning_rate", "optimizer"]
    )
    fig.show()
    fig2.show()

    df = study.trials_dataframe().drop(
        ["state", "datetime_start", "datetime_complete"], axis=1
    )

    print(f"SCORE: {scores}")
    print(f"Number of Finished Trials {len(study.trials)}")
    print(f"Number of Pruned Trials {len(pruned_trials)}")
    print(f"Number of Completed Trials {len(complete_trials)}")
    print(df)
