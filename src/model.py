import torch.nn as nn


class DeepNeuralNetwork(nn.Module):
    def __init__(self, n_features, n_targets, n_layers, hidden_size, dropout):
        super(DeepNeuralNetwork, self).__init__()
        layers = []
        for _ in range(n_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(n_features, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())

            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, n_targets))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
