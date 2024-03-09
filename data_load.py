import pydicom
from torch.utils.data import Dataset
from PIL import Image


# Define custom dataset class
class PneumoniaDataset(Dataset):
    def __init__(self, data_paths, labels, transforms=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        # Read the DICOM file
        dcm = pydicom.dcmread(path)
        # Extract the pixel array from the DICOM file
        img = dcm.pixel_array
        img = Image.fromarray(img)

        if self.transforms is not None:
            # Apply the specified transforms to the image
            img = self.transforms(img)

        # Extract the label
        label = self.labels[idx]
        return img, label