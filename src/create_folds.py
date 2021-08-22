import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    targets = df["is_canceled"].values

    # add kfold column for 5-Folds cross-validator
    skf = StratifiedKFold(n_splits=5)
    for fold, (train, valid) in enumerate(skf.split(X=df, y=targets)):
        df.loc[valid, "kfold"] = fold
        print(df["is_canceled"].value_counts())
    df.to_csv(config.TRAINING_FOLDS, index=False)
