import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch

class PneumomiaClassificationModel(pl.LightningModule):
    def __init__(self, weight=3):
        super().__init__()
        # Define the model architecture
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y = y.float().unsqueeze(1)  # Ensure target tensor has the same size as input tensor
        loss = self.loss_fn(outputs, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc(outputs.sigmoid(), y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y = y.float().unsqueeze(1)  # Ensure target tensor has the same size as input tensor
        loss = self.loss_fn(outputs, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.val_acc(outputs.sigmoid(), y), on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer