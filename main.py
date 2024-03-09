from model import PneumomiaClassificationModel
from resnet18_model import PneumoniaResnet18
import data_prep
import torchvision
from torchvision import transforms
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def CreateCheckpoint():
    # Create checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="./models",
        filename="best_model",
        save_top_k=1,
        mode="max"
    )
    resnet18_checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="./resnet18",
        filename="best_model",
        save_top_k=1,
        mode="max"
    )
    return checkpoint_callback, resnet18_checkpoint_callback

def _init_models():
    model = PneumomiaClassificationModel()
    resnet18_model = PneumoniaResnet18()
    return model, resnet18_model

def CreateTrainer(checkpoint_callback, resnet18_checkpoint_callback):
    # Create PyTorch Lighting Trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=TensorBoardLogger(save_dir="./logs"),
        log_every_n_steps=1,
        callbacks=checkpoint_callback,
        max_epochs=10
    )
    resnet18_trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=TensorBoardLogger(save_dir="./logs/restnet18"),
        log_every_n_steps=1,
        callbacks=resnet18_checkpoint_callback,
        max_epochs=35
    )
    return trainer, resnet18_trainer

def CreateEarlyStopping(trainer, resnet18_trainer):
    early_stopping = EarlyStopping(
        monitor="val_acc",
        min_delta=0,
        patience=10,
        verbose=True
    )
    resnet18_earlystopping = EarlyStopping(
        monitor="val_acc",
        min_delta=0,
        patience=10,
        verbose=True
    )
    return early_stopping, resnet18_earlystopping

def FitTrainer(trainer, train_loader, model, val_loader):
    trainer.fit(
        model,
        train_loader,
        val_loader
    )
    return trainer

def FitResNet18Trainer(resnet18_trainer, train_loader, resnet18_model, val_loader):
    resnet18_trainer.fit(
        resnet18_model,
        train_loader,
        val_loader)
    return resnet18_trainer

def LoadModel(checkpoint_callback, resnet18_checkpoint_callback):
    model = PneumomiaClassificationModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()
    model.to(device)

    resnet18_model = PneumoniaResnet18.load_from_checkpoint(resnet18_checkpoint_callback.best_model_path)
    resnet18_model.eval()
    resnet18_model.to(device)
    return model, resnet18_model

def EvaluateModel(model, val_dataset):
    preds, labels = [], []

    with torch.no_grad():
        for data, label in val_dataset:
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu())
            preds.append(pred)
            labels.append(label)
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()

        roc_auc = roc_auc_score(labels, preds)
        roc_auc_curve = roc_curve(labels, preds)
        plt.plot(roc_auc_curve[0])

        acc = torchmetrics.Accuracy(task="binary")(preds, labels)
        precision = torchmetrics.Precision(task="binary")(preds, labels)
        recall = torchmetrics.Recall(task="binary")(preds, labels)
        cm = torchmetrics.ConfusionMatrix(task="binary", num_class=2)(preds, labels)

    print(f"Val Accuracy: {acc}")
    print(f"Val Precision: {precision}")
    print(f"Val Recall: {recall}")
    print(f"Confusion Matrix {cm}")
    return acc, precision, recall

def EvaluateResNet18Model(resnet18_model, val_dataset):
    preds, labels = [], []

    with torch.no_grad():
        for data, label in val_dataset:
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(resnet18_model(data)[0].cpu())
            preds.append(pred)
            labels.append(label)
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()

        roc_auc = roc_auc_score(labels, preds)
        roc_auc_curve = roc_curve(labels, preds)

        acc = torchmetrics.Accuracy(task="binary")(preds, labels)
        precision = torchmetrics.Precision(task="binary")(preds, labels)
        recall = torchmetrics.Recall(task="binary")(preds, labels)
        cm = torchmetrics.ConfusionMatrix(task="binary", num_class=2)(preds, labels)

    print(f"Val Accuracy: {acc}")
    print(f"Val Precision: {precision}")
    print(f"Val Recall: {recall}")
    print(f"Confusion Matrix {cm}")
    return acc, precision, recall

def cam(model, img):
    # Run the model on the input image and get the predictions and features
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    # Reshape the features to a 2D tensor (512, 49) for matrix multiplication with weights
    features = features.reshape((512, 49))
    # Extract the weights of the last fully connected layer (output layer)
    weight_params = list(model.model.fc.parameters())[0]
    # Detach the weights to avoid gradients from being computed during CAM calculation
    weight = weight_params[0].detach()

    # Perform matrix multiplication between weights and features to get the CAM
    cam = torch.matmul(weight, features)
    # Reshape the CAM to match the size of the feature map (7x7) and move to CPU
    cam_img = cam.reshape(7, 7).cpu()
    return cam_img, torch.sigmoid(pred)

def visualize(img, cam, pred):
    # Move the input image and CAM tensor to CPU for visualization
    img = img[0].cpu()
    cam = torchvision.transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]

    # Create a figure with two subplots (original image and image with CAM)
    fig, axis = plt.subplots(1, 2)
    # Plot the original image in the first subplot
    axis[0].imshow(img, cmap="bone")
    # Plot the original image again in the second subplot
    axis[1].imshow(img, cmap="bone")
    # Overlay the CAM on the second subplot with alpha blending using the "jet" colormap
    axis[1].imshow(cam, alpha=0.5, cmap="jet")
    # Add a title to the second subplot indicating the predicted class based on the threshold (0.5)
    plt.title(pred > 0.5)
data = "Data/rsna-pneumonia-detection-challenge/stage_2_train_images/" #replace with actual location
labels = "Data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv" #replace with actual location

labels_df = data_prep.ReadLabels(labels)
train_transforms, val_transforms = data_prep.CreatePipelines()
train_dataset, val_dataset = data_prep.SeparateDataset(data, labels_df, train_transforms, val_transforms)
train_loader, val_loader = data_prep.CreateDataLoaders(train_dataset, val_dataset)

checkpoint_callback, resnet18_checkpoint_callback = CreateCheckpoint()
model, resnet18_model = _init_models()
trainer, resnet18_trainer = CreateTrainer(checkpoint_callback, resnet18_checkpoint_callback)
early_stopping, resnet18_earlystopping = CreateEarlyStopping(trainer, resnet18_trainer)
trainer = FitTrainer(trainer, train_loader, model, val_loader)
results = FitResNet18Trainer(resnet18_trainer, train_loader, resnet18_model, val_loader)
model, resnet18_model = LoadModel(checkpoint_callback, resnet18_checkpoint_callback)
acc, precision, recall = EvaluateModel(model, val_dataset)
acc, precision, recall = EvaluateResNet18Model(resnet18_model, val_dataset)


