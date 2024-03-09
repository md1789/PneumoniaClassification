import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch
import torchvision

class PneumoniaResnet18(pl.LightningModule):
    def __init__(self, weight=3):
        super().__init__()
        # Load the resnet18 model
        self.model = torchvision.models.resnet18()
        # Change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        # Change out_features of the last fully connected layer from 1000 to 1
        self.model.fc = torch.nn.Linear(
            in_features=512,
            out_features=1
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y = y.float().unsqueeze(1) # Ensure target tensor has the same size as the input tensor
        loss = self.loss_fn(outputs, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc(outputs.sigmoid(), y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y = y.float().unsqueeze(1) # Ensure target tensor has the same size as the input tensor
        loss = self.loss_fn(outputs, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_acc(outputs.sigmoid(), y), on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer