import torch.nn as nn


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def loss_fn(self, outputs, targets):
        return nn.BCELoss()(outputs, targets)

    def train(self, dataloader):
        # train model
        self.model.train()
        final_loss = 0
        for data in dataloader:
            self.optimizer.zero_grad()
            features = data["features"]
            targets = data["targets"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(dataloader)

    def evaluate(self, dataloader):
        # evaluation mode
        self.model.eval()
        final_loss = 0
        for data in dataloader:
            features = data["features"]
            targets = data["targets"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            final_loss += loss.item()
        return final_loss / len(dataloader)
