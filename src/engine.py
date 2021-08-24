import torch


class Engine:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def loss_fn(self, outputs, targets):
        # binary cross entroy
        return torch.nn.BCELoss()(outputs, targets)

    def train_fn(self, dataloader):
        """[Training the model on train_loader]
        Args:
            dataloader ([object]): [fetches data from a train_loader and serves the data up in batches]
        Returns:
            [int]: [target values and model's output values]
        """
        # set training mode
        self.model.train()
        final_targets, final_outputs = [], []
        for data in dataloader:
            # fetch values from HotelDataSet and convert to tensors
            features = data["features"]
            targets = data["targets"]
            outputs = self.model(features)
            loss = self.loss_fn(outputs, targets)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize + scheduler
            loss.backward()
            self.optimizer.step()
            # append to empty list
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
        # set evaluation mode for testing set
        self.model.eval()
        final_targets, final_outputs = [], []
        # disables gradient calculation
        with torch.no_grad():
            for data in dataloader:
                # fetch values from cutom dataset and convert to tensors
                features = data["features"]
                targets = data["targets"]
                outputs = self.model(features)
                # append to empty lists
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_outputs
