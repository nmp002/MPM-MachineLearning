######################## Description ############################
# Method 2 Script Used to Train the CNN on All Combinations of
# Image Inputs.
# This script handles one Fold distribution at a time. Change
# the specific train_ids and test_ids for each Fold.

# Created by Nicholas Powell
# Laboratory for Functional Optical Imaging & Spectroscopy
# University of Arkansas
#
# Please note: For easier reference, images are referred to
# as NADH, FAD, SHG, and ORR instead of I_755/blue, I_855/green,
# I_855/UV, and optical ratio, respectively.
#################################################################

## Imports ##
import torch
from torch.utils.data import Subset
import torchvision.transforms.v2 as tvt
from torch.utils.data import DataLoader
from scripts.microscopy_dataset import MicroscopyDataset
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from models.classification_CNN import classificationModel
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from collections import defaultdict
import os

# ==================================
# PRESETS/PARAMETERS CONFIGURATION
# ==================================
data_dir = "data/newData"
labels_csv = "data/newData/labels.csv"
label_fn = lambda x: torch.tensor(float(x > 25))

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()

# HYPERPARAMETERS
batch_size = 16
epochs = 996
learning_rate = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# TRANSFORMS
# ==================================
# Transformations for training set
train_transform = tvt.Compose([
    tvt.RandomVerticalFlip(p=0.25),
    tvt.RandomHorizontalFlip(p=0.25),
    tvt.RandomChoice([
        tvt.RandomRotation(degrees=[0, 0]),
        tvt.RandomRotation(degrees=[90, 90]),
        tvt.RandomRotation(degrees=[180, 180]),
    ]),
    tvt.RandomResizedCrop(size=512, scale=(0.8, 1.0))
])


# ==================================
# RESULTS FILE CONFIGURATION
# ==================================
# Get current date and time
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H:%M")
results_file = f"results_{timestamp}.md"
with open(results_file, 'w') as f:
    f.write('# Results \n\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Epochs: {epochs}\n')
    f.write(f'Learning rate: {learning_rate}\n')




# Function to get indices of img_labels belonging to a given set of sample_ids
def get_indices_by_sample_ids(img_labels, sample_ids_set):
    indices = []
    for idx, (sample_dir, fov_dir, label, sample_id) in enumerate(img_labels):
        if sample_id in sample_ids_set:
            indices.append(idx)
    return indices

# =========================================
# INITIALIZE MODELS, OPTIMIZERS, & LOSS FNS
# =========================================

# ORR maps excluded from channels for now -- too many NaN values
channel_set = [
    ['nadh'],   # Model 1
    ['fad'],    # Model 2
    ['shg'],    # Model 3
    ['orr'],    # Model 4

    ['nadh', 'fad'],    # Model 5
    ['nadh', 'shg'],    # Model 6
    ['nadh', 'orr'],    # Model 7
    ['fad', 'shg'],     # Model 8
    ['fad', 'orr'],     # Model 9
    ['shg', 'orr'],     # Model 10

    ['nadh', 'fad', 'shg'], # Model 11
    ['nadh', 'fad', 'orr'], # Model 12
    ['nadh', 'shg', 'orr'], # Model 13
    ['fad', 'shg', 'orr'],  # Model 14

    ['nadh', 'fad', 'shg', 'orr']   # Model 15
]

models = []
optimizers = []
loss_fns = []
datasets = []
train_datasets = []
train_loaders = []
test_loaders = []

train_ids = ['Sample_023', 'Sample_004', 'Sample_010', 'Sample_025', 'Sample_014', 'Sample_027',
             'Sample_003', 'Sample_030', 'Sample_013', 'Sample_018', 'Sample_028', 'Sample_011',
             'Sample_006', 'Sample_022', 'Sample_016', 'Sample_012', 'Sample_007', 'Sample_001',
             'Sample_015', 'Sample_009', 'Sample_029', 'Sample_019', 'Sample_008']

# Change training and test ids for each Fold distribution
# test_ids = ['Sample_023', 'Sample_004', 'Sample_010', 'Sample_025', 'Sample_014', 'Sample_027']    # Fold 1 - (High/Low = 3/3)
test_ids = ['Sample_002', 'Sample_026', 'Sample_017', 'Sample_024', 'Sample_005', 'Sample_020']  # Fold 2 - (High/Low = 3/3)
# test_ids = ['Sample_003', 'Sample_030', 'Sample_013', 'Sample_018', 'Sample_028', 'Sample_011']  # Fold 3 - (High/Low = 3/3)
# test_ids = ['Sample_006','Sample_022', 'Sample_016', Sample_012', 'Sample_007', 'Sample_001']     # Fold 4 - (High/Low = 3/3)
# test_ids = ['Sample_015', 'Sample_009', 'Sample_029', 'Sample_019', 'Sample_008']     # Fold 5 - (High/Low = 2/3)

for channels in channel_set:
    in_channels = len(channels)
    model = classificationModel(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.BCELoss()

    dataset = MicroscopyDataset(
        data_dir=data_dir,
        labels_csv=labels_csv,
        channels=channels,
        transform=None,
        label_fn=label_fn
    )

    train_dataset = MicroscopyDataset(
        data_dir=data_dir,
        labels_csv=labels_csv,
        channels=channels,
        transform=train_transform,
        label_fn=label_fn
    )

    # Create lists of indices for each split
    train_indices = get_indices_by_sample_ids(train_dataset.img_labels, set(train_ids))
    test_indices = get_indices_by_sample_ids(dataset.img_labels, set(test_ids))

    # Create dataset subsets
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    models.append(model)
    optimizers.append(optimizer)
    loss_fns.append(loss_fn)
    datasets.append(dataset)
    train_datasets.append(train_dataset)
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)


# ==================================
# TRAINING LOOP
# ==================================

for i in range(len(models)):
    with open(results_file, 'a') as f:
        f.write(f'\nTraining model_{i+1}:\n')
        f.write(f'Channel Inputs: {channel_set[i]}\n')

    for epoch in range(epochs):
        if (epoch+1) == 1 or (epoch+1) % 50 == 0:
            with open(results_file, 'a') as f:
                f.write(f'Epoch {epoch+1}/{epochs}\n')

        model = models[i]
        optimizer = optimizers[i]
        loss_fn = loss_fns[i]

        # Train the model
        model.train()
        for x, target, _ in train_loaders[i]:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze()

            invalid_outs = out[(out < 0) | (out > 1)]
            if invalid_outs.numel() > 0 or torch.isnan(out).any():
                print(f'Found invalid model outputs: {invalid_outs}')
            invalid_targets = target[(target < 0) | (target > 1)]
            if invalid_targets.numel() > 0:
                print(f'Found invalid model targets: {invalid_targets}')


            loss = loss_fn(out, target)

            loss.backward()
            optimizer.step()

        # Test the model at the final epoch
        if (epoch+1) == epochs:
            with open(results_file, 'a') as f:
                f.write(f'\nTesting model_{i+1}...\n')

            # Save the model used for testing
            torch.save(model.state_dict(), f'model_{i+1}_epoch{epoch+1}.pt')

            with torch.no_grad():
                model.eval()
                ys, targets = [], []
                sample_targets = defaultdict(list)
                for img, target, sample_ids in test_loaders[i]:
                    img = img.to(device)
                    y = model(img).squeeze().cpu()
                    y = y.numpy().astype(np.float64).tolist()

                    for id_name, target_name in zip(sample_ids, target):
                        sample_targets[id_name].append(target_name.item())

                    target = target.cpu()
                    target = target.numpy().astype(np.float64).tolist()
                    ys.append(y)
                    targets.append(target)

                averaged_targets = {}
                for sample_id, target_list in sample_targets.items():
                    if all(target == target_list[0] for target in target_list):
                        averaged_targets[sample_id] = target_list[0]
                    else:
                        print(f'Warning: Inconsistent targets for {sample_id} -> {target_list}')
                        epoch = epochs

                ys = [item for y in ys for item in y]
                sample_ys = np.mean(np.array(ys).reshape(-1, 5), axis=1)

                threshold = 0.50
                true_values = list(averaged_targets.values())

                predictions = [out_value >= threshold for out_value in sample_ys]

                # Compare predictions with true values
                mismatched_samples = [sample for sample, pred, true in zip(averaged_targets.keys(),
                                        predictions, true_values) if pred != true]

                # Log mismatched samples
                with open(results_file, 'a') as f:
                    f.write('Incorrect Predictions:\n')
                    for sample in mismatched_samples:
                        f.write(f' - {sample}\n')
                print(f'Incorrect Predictions: {mismatched_samples}')   # Log to console as well

                # Plot confusion matrix
                conf_matrix = ConfusionMatrixDisplay.from_predictions(true_values, predictions)
                conf_matrix.plot()
                plt.savefig(f'Model_{i + 1}_epoch{epoch + 1}_confusion_matrix.png')


