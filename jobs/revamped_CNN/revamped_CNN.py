## Imports ##
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torchvision.transforms.v2 as tvt
from torch.utils.data import DataLoader
from scripts.microscopy_dataset import MicroscopyDataset
import random
import datetime
import matplotlib.pyplot as plt
import os
from models.classification_CNN import classificationModel

# ==================================
# PRESETS/PARAMETERS CONFIGURATION
# ==================================
channels = ['nadh','shg']
in_channels = len(channels)
data_dir = "data/newData"
labels_csv = "data/newData/labels.csv"
label_fn = lambda x: torch.tensor(float(x > 25))

torch.manual_seed(42)
random.seed(42)

# HYPERPARAMETERS
batch_size = 16
epochs = 250
learning_rate = 1e-6

# SET SPLITS
val_split = 0.2
test_split = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# TRANSFORMS
# ==================================
# Transformations for training set
train_transform = tvt.Compose([
    tvt.RandomVerticalFlip(p=0.25),
    tvt.RandomHorizontalFlip(p=0.25),
    tvt.RandomRotation(degrees=(-180, 180))])

# ==================================
# RESULTS FILE CONFIGURATION
# ==================================
# Get current date and time
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H:%M")
results_file = f"results_{timestamp}.md"
with open(results_file, 'w') as f:
    f.write('Results \n\n')
    f.write(f'Channels used: {channels}\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Epochs: {epochs}\n')
    f.write(f'Learning rate: {learning_rate}\n\n')

# ==================================
# LOAD FULL DATASET
# ==================================
dataset = MicroscopyDataset(
    data_dir=data_dir,
    labels_csv=labels_csv,
    channels=channels,
    transform=None,
    label_fn=label_fn
)

# ==================================
# IMAGE CHECKPOINT
# ==================================
# Plot 5 random samples to ensure images are loaded correctly from dataset
total_samples = len(dataset)
random_indices = random.sample(range(total_samples), 5)
for i, idx in enumerate(random_indices):
    image, label, sample_id = dataset[idx]
    fig, ax = plt.subplots(1, in_channels)
    if in_channels == 1:
        ax = [ax]
    for channel in range(in_channels):
        channel_image = image[channel].numpy()
        ax[channel].imshow(channel_image, cmap='gray')
        ax[channel].set_title(f'Channel {channel+1} - Sample: {sample_id}\nLabel: {label.item()}')
        ax[channel].axis('off')
    plt.tight_layout()

    plot_filename = f'{sample_id}_Index_{idx}.png'
    plt.savefig(plot_filename)
    plt.close(fig)

    with open(results_file, 'a') as f:
        f.write(f'## {sample_id} (Label: {label.item()})\n\n')
        f.write(f'![{sample_id}]({plot_filename})\n\n')

    if os.path.exists(plot_filename):
        os.remove(plot_filename)
    else:
        print(f'Warning: {plot_filename} does not exist')

# ==================================
# LOAD SEPARATE TRAINING DATASET
# ==================================
train_dataset = MicroscopyDataset(
    data_dir=data_dir,
    labels_csv=labels_csv,
    channels=channels,
    transform=train_transform,
    label_fn=label_fn
)

# ==================================
# SAMPLE-BASED SPLITTING
# ==================================
# Grab all unique sample directories from img_labels
sample_ids = list(set([sample_id for sample_dir, fov_dir, label, sample_id in dataset.img_labels]))

# Split the sample_ids into train, val, and test sets
train_ids, temp_ids = train_test_split(
    sample_ids,
    test_size = (val_split + test_split),
    random_state=42,
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size = test_split / (val_split + test_split),
    random_state=42,
)

# Function to get indices of img_labels belonging to a given set of sample_ids
def get_indices_by_sample_ids(img_labels, sample_ids_set):
    indices = []
    for idx, (sample_dir, fov_dir, label, sample_id) in enumerate(img_labels):
        if sample_id in sample_ids_set:
            indices.append(idx)
    return indices

# Create lists of indices for each split
train_indices = get_indices_by_sample_ids(train_dataset.img_labels, set(train_ids))
val_indices = get_indices_by_sample_ids(dataset.img_labels, set(val_ids))
test_indices = get_indices_by_sample_ids(dataset.img_labels, set(test_ids))

with open(results_file, 'a') as f:
    f.write(f'Training Indices ({len(train_indices)}): {train_indices}\n')
    f.write(f'Validation Indices ({len(val_indices)}): {val_indices}\n')
    f.write(f'Test Indices ({len(test_indices)}): {test_indices}\n')

# Create dataset subsets
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# ==================================
# DATALOADERS
# ==================================
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

