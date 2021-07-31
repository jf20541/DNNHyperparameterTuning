import torch.nn as nn


class Model(nn.Module):
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(nfeatures, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Sigmoid())

            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden_size, ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
