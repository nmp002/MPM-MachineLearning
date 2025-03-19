## Description ##
# This is a from-scratch dataset for my Highlands Machine Learning Project (Master's Thesis)
# Created on: March 18th, 2025

## Imports ##
import os
import pandas as pd
from torch.utils.data import Dataset
import tifffile
import torch
from pathlib import Path

def tiff_to_tensor(path):
    img = tifffile.imread(path) # Returns a numpy array
    img_tensor = torch.from_numpy(img).float()  # Convert NumPy array to tensor
    if img_tensor.dim() == 2:  # If grayscale, add channel dimension
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


class MicroscopyDataset(Dataset):
    def __init__(self, data_dir ,labels_csv, channels=('fad','nadh','orr','shg'), transform=None, label_fn=None):
        """
        data_dir: Root directory that contains all sample directories
        sample_labels: A list with the mapping of sample to label (recurrence score)
        transform: Optional transforms for images
        channels: List of channel names
        """

        self.data_dir = data_dir
        self.sample_labels = pd.read_csv(labels_csv)
        self.sample_labels.set_index('sample_id', inplace=True)
        self.transform = transform
        self.channels = channels
        self.label_fn = label_fn

        # Automatically index all (sample_dir, fov_dir)
        self.img_labels = []

        sample_dirs = sorted([os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_path in sample_dirs:
            sample_id = Path(sample_path).name
            fov_dirs = sorted([f for f in os.listdir(sample_path)
                               if os.path.isdir(os.path.join(sample_path, f))])

            for fov_dir in fov_dirs:
                # Assign label from labels_csv
                label = torch.tensor(self.sample_labels.loc[Path(sample_path).name, 'recurrence_score'])

                self.img_labels.append((sample_path, fov_dir, label, sample_id))

        print(f"Found {len(sample_dirs)} Samples for a total of {len(self.img_labels)} fovs.")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        sample_dir, fov_dir, label, sample_id = self.img_labels[idx]

        # Path to the specific fov directory
        fov_path = os.path.join(self.data_dir, sample_dir, fov_dir)

        # List to collect the tensors for each selected channel
        channel_tensors = []

        # Iterate over the channels specified during initialization
        for channel in self.channels:
            # Build the file path dynamically
            image_filename = f'{channel}.tiff'
            image_path = os.path.join(fov_path, image_filename)

            # Read the image and append to the list
            img_tensor = tiff_to_tensor(image_path)
            channel_tensors.append(img_tensor)

        # Concatenate along the channel dimension
        image = torch.cat(channel_tensors, dim=0)

        # Apply optional transforms
        if self.transform:
            image = self.transform(image)

        label = self.label_fn(label)

        return image, label, sample_id





