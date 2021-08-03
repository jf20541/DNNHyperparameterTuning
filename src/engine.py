import torch


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    # binary cross entroy (1, 0)
    def loss_fn(self, outputs, targets):
        return torch.nn.BCELoss()(outputs, targets)

    def train_fn(self, dataloader):
        """[Training the model on train_loader]
        Args:
            dataloader ([object]): [fetches data from a train_loader and serves the data up in batches]
        Returns:
            [int]: [target values and model's output values]
        """
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

    def eval_fn(self, dataloader):
        """[Evaluate the model on eval_loader]
        Args:
            dataloader ([object]): [fetches data from a test_loader and serves the data up in batches]
        Returns:
            [int]: [target values and model's output values]
        """
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
