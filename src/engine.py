import torch
import torch.nn as nn


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(outputs, targets):
        return nn.BCELoss()(outputs, targets.view(-1, 1))

    def train_fn(self, dataloader):
        # train mode
        self.model.train()
        final_loss = 0
        for data in dataloader:
            self.optimizer.zero_grad()
            features = data["x"]
            targets = data["y"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(dataloader)

    def eval_fn(self, dataloader):
        # evaluation mode
        self.model.eval()
        final_loss = 0
        for data in dataloader:
            features = data["x"]
            targets = data["y"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            final_loss += loss.item()
        return final_loss / len(dataloader)
