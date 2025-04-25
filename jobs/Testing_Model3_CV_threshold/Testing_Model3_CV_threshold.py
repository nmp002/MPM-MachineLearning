## Description ##
# Test the 4 left-out samples from CV on the calculated average threshold.
#
#
# The 5 thresholds from each fold were averaged together to get a final
# threshold that will be used to test the 4 test samples.
#
#
# Test Set: Samples 10-13 (High/Low = 2/2)

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
import pandas as pd

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
epochs = 1000
learning_rate = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# TRANSFORMS
# ==================================
# Transformations for training set
train_transform = tvt.Compose([
    tvt.RandomVerticalFlip(p=0.25),
    tvt.RandomHorizontalFlip(p=0.25),
    tvt.RandomRotation(degrees=(-180, 180)),
    tvt.RandomResizedCrop(size=512, scale=(0.8, 1.0))  # Random zoom-in
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

channels = ['fad']   # Model 3


train_ids = ['Sample_001', 'Sample_002', 'Sample_003','Sample_004','Sample_014',
            'Sample_005', 'Sample_006', 'Sample_007', 'Sample_008', 'Sample_009',
            'Sample_015', 'Sample_016', 'Sample_017', 'Sample_018', 'Sample_019',
            'Sample_020', 'Sample_022', 'Sample_023', 'Sample_024', 'Sample_025',
            'Sample_026', 'Sample_027', 'Sample_028', 'Sample_029', 'Sample_030']

test_ids = ['Sample_010', 'Sample_011', 'Sample_012', 'Sample_013']



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


# ==================================
# TRAINING LOOP
# ==================================


with open(results_file, 'a') as f:
    f.write(f'\nTraining model_2:\n')
    f.write(f'Channel Inputs: {channels}\n')

train_losses = []

for epoch in range(epochs):
    if (epoch+1) == 1 or (epoch+1) % 50 == 0:
        with open(results_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{epochs}\n')

    # Train the model
    model.train()
    epoch_train_loss = 0.0
    for x, target, _ in train_loader:
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
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)


    if (epoch+1) == epochs:
        torch.save(model.state_dict(), 'Model2_for_Testing.pt')
        with open(results_file, 'a') as f:
            f.write('Testing Model 2 with Average CV Threshold...\n')

        model.load_state_dict(torch.load('Model2_for_Testing.pt'))


        with torch.no_grad():
            model.eval()
            ys, targets = [], []
            sample_targets = defaultdict(list)
            for img, target, sample_ids in test_loader:
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
            threshold = 0.57526
            with open(results_file, 'a') as f:
                f.write(f'Average CV Threshold for Model 2: {threshold}\n')

            t = list(averaged_targets.values())
            o = sample_ys
            preds = [out_value >= threshold for out_value in o]
            conf_matrix = ConfusionMatrixDisplay.from_predictions(t, preds)
            conf_matrix.plot()
            plt.savefig(f'Model_2_epoch{epochs}_confusion_matrix_Test_Set.png')



    # Plot loss curves at specified epochs
    if (epoch + 1) == epochs:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot raw training loss
        ax.plot(train_losses, label='Training Loss (Raw)', color='blue', alpha=0.3)

        # smoothed training loss using a moving average
        smoothed_train = pd.Series(train_losses).rolling(window=30, min_periods=1).mean()
        ax.plot(smoothed_train, label='Training Loss (Smoothed)', color='blue', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Model 2 Loss Curve')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'model_2_loss_epoch{epoch + 1}.png')
        plt.close(fig)



