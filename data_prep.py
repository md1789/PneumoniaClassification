import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from data_load import PneumoniaDataset

batch_size = 32
num_workers = 4

def ReadLabels(labels):
    labels_df = pd.read_csv(labels)
    labels_df.shape
    labels_df.head()
    labels_df = labels_df[["patientId", "Target"]]
    labels_df.head()
    return labels_df

def CreatePipelines():
    # Data transformation pipelines for data preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
        transforms.RandomResizedCrop((224, 224), scale=(0.25, 1), antialias=True)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    return train_transforms, val_transforms

def SeparateDataset(data, labels_df, train_transforms, val_transforms):
    # Split the labels into train and validation sets
    train_labels, val_labels = train_test_split(labels_df.values, test_size=0.1, random_state=42)

    # Get the paths for train and validation images
    train_paths = [os.path.join(data, image[0] + ".dcm") for image in train_labels]
    val_paths = [os.path.join(data, image[0] + ".dcm") for image in val_labels]

    # Create train dataset
    train_dataset = PneumoniaDataset(train_paths, train_labels[:, 1], transforms=train_transforms)

    # Create validation dataset
    val_dataset = PneumoniaDataset(val_paths, val_labels[:, 1], transforms=val_transforms)
    return train_dataset, val_dataset

def CreateDataLoaders(train_dataset, val_dataset):
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return train_loader, val_loader
