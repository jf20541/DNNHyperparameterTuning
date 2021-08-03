import torch


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    # binary cross entroy (1, 0)
    def loss_fn(self, outputs, targets):
        return torch.nn.BCELoss()(outputs, targets)

    # training function for the train_loader
    def train_fn(self, dataloader):
        self.model.train()
        final_targets, final_outputs = [], []
        for data in dataloader:
            features = data["features"]
            targets = data["targets"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_outputs

    # evaluation function for the test_loader
    def eval_fn(self, dataloader):
        self.model.eval()
        final_targets, final_outputs = [], []
        with torch.no_grad():
            for data in dataloader:
                features = data["features"]
                targets = data["targets"]
                outputs = self.model(features)
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_outputs
