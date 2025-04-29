######################## Description ############################
# Custom Dataset to Manage and Load Multi-Channel MPM Image
# Data for Deep Learning.

# Created by Nicholas Powell
# Laboratory for Functional Optical Imaging & Spectroscopy
# University of Arkansas
#
# Please note: For easier reference, images are referred to
# as NADH, FAD, SHG, and ORR instead of I_755/blue, I_855/green,
# I_855/UV, and optical ratio, respectively.
#################################################################

## Imports ##
import os
import pandas as pd
import torch
import tifffile
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# Function to load a .tiff image file and convert it into a PyTorch tensor
def tiff_to_tensor(path):
    """
    Read a .tiff file, convert it to a PyTorch tensor, and preprocess it.

    Parameters:
    - path (str): Path to the .tiff image file.

    Returns:
    - torch.Tensor: Tensor representation of the image.
    """
    img = tifffile.imread(path)  # Read the .tiff image into a NumPy array
    img_tensor = torch.from_numpy(img).float()  # Convert the NumPy array to a tensor (float)

    # Replace NaN and infinite values with reasonable defaults (useful for invalid pixel data)
    img_tensor = torch.nan_to_num(img_tensor, nan=1.0, posinf=1.0, neginf=0.0)

    # If the image is grayscale (2D), add a channel dimension (to make it 3D: channels x height x width)
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

# Define a PyTorch Dataset class to manage microscopy images and labels
class MicroscopyDataset(Dataset):
    def __init__(self, data_dir, labels_csv, channels=('fad', 'nadh', 'orr', 'shg'), transform=None, label_fn=None):
        """
        Custom dataset initialization.

        Parameters:
        - data_dir (str): Root directory containing sample directories. Each subdirectory corresponds
                          to one sample.
        - labels_csv (str): Path to the CSV file containing sample IDs and their recurrence scores (labels).
        - channels (tuple): Tuple of strings specifying channel names (e.g., 'fad', 'nadh').
        - transform (callable, optional): Transformations to apply to the image data.
        - label_fn (callable, optional): Function to process the labels (e.g., convert to one-hot or normalize).
        """
        self.data_dir = data_dir
        self.sample_labels = pd.read_csv(labels_csv)
        self.sample_labels.set_index('sample_id', inplace=True)
        self.transform = transform
        self.channels = channels
        self.label_fn = label_fn

        # List to store all sample and field-of-view (FoV) metadata
        self.img_labels = []

        # Iterate through all sample directories in the given data directory
        sample_dirs = sorted([os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_path in sample_dirs:
            sample_id = Path(sample_path).name

            fov_dirs = sorted([f for f in os.listdir(sample_path)
                               if os.path.isdir(os.path.join(sample_path, f))])

            for fov_dir in fov_dirs:
                label = torch.tensor(self.sample_labels.loc[sample_id, 'recurrence_score'])
                self.img_labels.append((sample_path, fov_dir, label, sample_id))

        print(f"Found {len(sample_dirs)} Samples for a total of {len(self.img_labels)} FoVs.")

        # === Compute 99.5th percentile values for NADH, FAD, and SHG across the dataset === #
        self.channel_percentile = {"nadh": 0.0, "fad": 0.0, "shg": 0.0}
        pixel_values = {"nadh": [], "fad": [], "shg": []}

        print("Collecting pixel values for percentile normalization...")
        for sample_path, fov_dir, _, sample_id in self.img_labels:
            fov_path = os.path.join(self.data_dir, sample_id, fov_dir)
            for channel in self.channels:
                if channel in pixel_values:
                    image_path = os.path.join(fov_path, f"{channel}.tiff")
                    img_tensor = tiff_to_tensor(image_path)
                    pixel_values[channel].append(img_tensor.flatten())

        # Compute 99.5th percentile intensity for each relevant channel
        for channel in pixel_values:
            if pixel_values[channel]:  # Only compute if there are images loaded
                all_pixels = torch.cat(pixel_values[channel]).numpy()
                self.channel_percentile[channel] = np.percentile(all_pixels, 99.5)
            else:
                # If no images loaded for this channel, skip normalization
                print(f"Warning: No pixel values collected for {channel}. Skipping percentile calculation.")
                self.channel_percentile[channel] = None

        print(f"Channel 99.5th percentiles computed: {self.channel_percentile}")

    def __len__(self):
        """
        Return the total number of field-of-views (FoVs) in the dataset.

        Returns:
        - int: Number of FoVs in the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Get the data sample (image and label) at the specified index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (image, label, sample_id)
            - image (torch.Tensor): Multi-channel image tensor with channels specified in `self.channels`.
            - label (torch.Tensor): Label tensor (processed by `label_fn` if provided).
            - sample_id (str): Sample ID, useful for debugging or traceability.
        """
        sample_dir, fov_dir, label, sample_id = self.img_labels[idx]
        fov_path = os.path.join(self.data_dir, sample_id, fov_dir)

        channel_tensors = []

        for channel in self.channels:
            image_filename = f'{channel}.tiff'
            image_path = os.path.join(fov_path, image_filename)

            img_tensor = tiff_to_tensor(image_path)

            # Normalize NADH, FAD, SHG using precomputed 99.5th percentiles
            if channel in self.channel_percentile and self.channel_percentile[channel] is not None:
                pval = self.channel_percentile[channel]
                if pval > 0:
                    img_tensor = img_tensor / pval
                    img_tensor = torch.clamp(img_tensor, 0, 1)

            # ORR maps remain unchanged (already between 0 and 1)
            channel_tensors.append(img_tensor)

        image = torch.cat(channel_tensors, dim=0)

        if self.transform:
            image = self.transform(image)

        label = self.label_fn(label)

        return image, label, sample_id
