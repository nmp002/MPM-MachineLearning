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
from torch.utils.data import Dataset
import tifffile
import torch
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

        Attributes:
        - self.data_dir: Root directory for the data.
        - self.sample_labels: DataFrame containing the mapping between samples and labels (indexed by sample ID).
        - self.transform: Transformations to apply to the images (if any).
        - self.channels: Channels to load for each image (e.g., FAD, NADH).
        - self.label_fn: Function to preprocess the labels.
        - self.img_labels: List containing metadata for each image (sample directory, fov directory, label, sample ID).
        """
        self.data_dir = data_dir
        self.sample_labels = pd.read_csv(labels_csv)  # Load CSV file as a DataFrame
        self.sample_labels.set_index('sample_id', inplace=True)  # Use sample_id as the index for efficient lookups
        self.transform = transform
        self.channels = channels
        self.label_fn = label_fn

        # List to store all sample and field-of-view (FoV) metadata
        self.img_labels = []

        # Iterate through all sample directories in the given data directory
        sample_dirs = sorted([os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_path in sample_dirs:
            sample_id = Path(sample_path).name  # Extract sample ID from the directory name

            # Get all field-of-view directories within the current sample directory
            fov_dirs = sorted([f for f in os.listdir(sample_path)
                               if os.path.isdir(os.path.join(sample_path, f))])

            for fov_dir in fov_dirs:
                # Assign a label from the CSV file (e.g., recurrence_score)
                label = torch.tensor(self.sample_labels.loc[Path(sample_path).name, 'recurrence_score'])

                # Append metadata: (sample directory, fov directory, label, and sample ID)
                self.img_labels.append((sample_path, fov_dir, label, sample_id))

        # Print details about the loaded dataset for debugging or progress tracking
        print(f"Found {len(sample_dirs)} Samples for a total of {len(self.img_labels)} FoVs.")

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
        # Retrieve metadata for the specified index
        sample_dir, fov_dir, label, sample_id = self.img_labels[idx]

        # Path to the specific FoV directory
        fov_path = os.path.join(self.data_dir, sample_id, fov_dir)

        # List to store tensors for each channel
        channel_tensors = []

        # Process each channel and load the corresponding image
        for channel in self.channels:
            # Dynamically construct the image file path based on channel name
            if channel in ['fad', 'nadh', 'shg']:  # Use normalized versions for these channels
                image_filename = f'{channel}_norm.tiff'
            else:  # Keep the original filename for 'orr' and other potential channels
                image_filename = f'{channel}.tiff'

            image_path = os.path.join(fov_path, image_filename)

            # Load the image and append its tensor to the list
            img_tensor = tiff_to_tensor(image_path)
            channel_tensors.append(img_tensor)

        # Concatenate along the channel dimension to create a single multi-channel image tensor
        image = torch.cat(channel_tensors, dim=0)

        # Apply any user-defined transformations (if provided)
        if self.transform:
            image = self.transform(image)

        # Process the label with the given function (e.g., scaling, normalization, etc.)
        label = self.label_fn(label)

        # Return the processed image tensor, label tensor, and the sample ID
        return image, label, sample_id





