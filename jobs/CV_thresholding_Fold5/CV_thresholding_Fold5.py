## Description ##
# Performing cross-validation on 25 of the 29 samples,
# leaving out 4 samples for final testing. For each of the 5 folds,
# the threshold will be set based on the validation set. Then,
# the 5 thresholds will be averaged together to get a final
# threshold that will be used to test the 4 test samples.
# This script is for Fold 5 validation, training on Folds 1-4.
# Fold 1: Samples 1-4, 14 (High/Low = 3/2)
# Fold 2: Samples 5-9 (High/Low = 2/3)
# Fold 3: Samples 15-19 (High/Low = 2/3)
# Fold 4: Samples 20, 22-25 (High/Low = 3/2)
# Fold 5: Samples 26-30 (High/Low = 2/3)
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
epochs = 3000   # Max number of epochs if there is no early stopping
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


# ==================================
# DEFINE SCORING FUNCTION
# ==================================
# Function to calculate scores and plot roc curve/confusion matrix
def score_em(t, o, which_set):
    fpr, tpr, thresholds = roc_curve(t, o)
    test_score = auc(fpr, tpr)
    thresh = thresholds[np.argmax(tpr - fpr)]
    preds = [out_value >= thresh for out_value in o]
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=test_score)
    roc_display.plot()
    plt.savefig(f'Model_{i+1}_epoch{best_epoch + 1}_roc_curve_{which_set}.png')
    conf_matrix = ConfusionMatrixDisplay.from_predictions(t, preds)
    conf_matrix.plot()
    plt.savefig(f'Model_{i+1}_epoch{best_epoch + 1}_confusion_matrix_{which_set}.png')

    return thresh


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
val_loaders = []

train_ids = ['Sample_001', 'Sample_002', 'Sample_003','Sample_004','Sample_014',
 'Sample_005', 'Sample_006', 'Sample_007', 'Sample_008', 'Sample_009', 'Sample_015', 'Sample_016', 'Sample_017', 'Sample_018', 'Sample_019',
 'Sample_020', 'Sample_022', 'Sample_023', 'Sample_024', 'Sample_025']

# val_ids = ['Sample_001', 'Sample_002', 'Sample_003','Sample_004','Sample_014']      # Fold 1
# val_ids = ['Sample_005', 'Sample_006', 'Sample_007', 'Sample_008', 'Sample_009']  # Fold 2
# val_ids = ['Sample_015', 'Sample_016', 'Sample_017', 'Sample_018', 'Sample_019']  # Fold 3
# val_ids = ['Sample_020', 'Sample_022', 'Sample_023', 'Sample_024', 'Sample_025']  # Fold 4
# val_ids = ['Sample_026', 'Sample_027', 'Sample_028', 'Sample_029', 'Sample_030']  # Fold 5
val_ids = ['Sample_026', 'Sample_027', 'Sample_029', 'Sample_030']  # Fold 5 - sitting out Sample 28
# Leaving out Samples 10-13 for future testing

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
    val_indices = get_indices_by_sample_ids(dataset.img_labels, set(val_ids))

    # Create dataset subsets
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    models.append(model)
    optimizers.append(optimizer)
    loss_fns.append(loss_fn)
    datasets.append(dataset)
    train_datasets.append(train_dataset)
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)


# ==================================
# TRAINING LOOP
# ==================================

for i in range(len(models)):
    with open(results_file, 'a') as f:
        f.write(f'\nTraining model_{i+1}:\n')
        f.write(f'Channel Inputs: {channel_set[i]}\n')

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 150
    patience_counter = 0
    early_stopping = False
    best_model_path = f'model_{i+1}_best.pt'

    for epoch in range(epochs):
        model = models[i]
        optimizer = optimizers[i]
        loss_fn = loss_fns[i]

        # Train the model
        model.train()
        epoch_train_loss = 0.0
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
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        ## Validation to monitor for overfitting ##
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x, target, _ in val_loaders[i]:
                x, target = x.to(device), target.to(device)
                out = model(x).squeeze()
                loss = loss_fn(out, target)
                epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_dataset)
            val_losses.append(avg_val_loss)

            # Early stopping logic:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                best_epoch = epoch
                with open(results_file, 'a') as f:
                    f.write(f"✅ Epoch {epoch + 1}: Validation loss improved to {avg_val_loss:.6f}. Saving model.\n")
            else:
                patience_counter += 1
                with open(results_file, 'a') as f:
                    f.write(
                        f"🔁 Epoch {epoch + 1}: No improvement in validation loss. Patience counter: {patience_counter}/{patience}\n")
                if patience_counter >= patience:
                    with open(results_file, 'a') as f:
                        f.write(f"⏹️ Early stopping triggered at Epoch {epoch + 1}.\n")

                        f.write(f'\nTesting Model_{i + 1}:\n')

                    early_stopping = True
                    model.load_state_dict(torch.load(best_model_path))

                    # Test the validation fold once training is complete and calculate Fold 5 threshold
                    with open(results_file, 'a') as f:
                        f.write(f'\nTesting model_{i+1} on Fold 5...\n')

                    with torch.no_grad():
                        model.eval()
                        ys, targets = [], []
                        sample_targets = defaultdict(list)
                        for img, target, sample_ids in val_loaders[i]:
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
                        threshold = score_em(list(averaged_targets.values()), sample_ys, 'Validation')
                        with open(results_file, 'a') as f:
                            f.write(f'Fold 5 Validation Threshold: {threshold}\n')

                    ## Test the training set as well
                    with open(results_file, 'a') as f:
                        f.write(f'\nTesting model_{i+1} on Training Folds (1-4)...\n')
                    with torch.no_grad():
                        model.eval()
                        ys_train, targets_train = [], []
                        sample_targets_train = defaultdict(list)
                        for img, target, sample_ids in train_loaders[i]:
                            img = img.to(device)
                            y = model(img).squeeze().cpu()
                            y = y.numpy().astype(np.float64).tolist()

                            for id_name, target_name in zip(sample_ids, target):
                                sample_targets_train[id_name].append(target_name.item())

                            target = target.cpu()
                            target = target.numpy().astype(np.float64).tolist()
                            ys_train.append(y)
                            targets_train.append(target)

                        averaged_targets_train = {}
                        for sample_id, target_list in sample_targets_train.items():
                            if all(target == target_list[0] for target in target_list):
                                averaged_targets_train[sample_id] = target_list[0]
                            else:
                                print(f'Warning: Inconsistent targets for {sample_id} -> {target_list}')
                                epoch = epochs

                        ys_train = [item for y in ys_train for item in y]
                        sample_ys_train = np.mean(np.array(ys_train).reshape(-1, 5), axis=1)
                        threshold_train = score_em(list(averaged_targets_train.values()), sample_ys_train, 'Training')
                        with open(results_file, 'a') as f:
                            f.write(f'Fold 5 Training Threshold: {threshold_train}\n')

                    # Plot the final train/val loss curve
                    fig, ax = plt.subplots(figsize=(6, 4))

                    # Plot raw training loss
                    ax.plot(train_losses, label='Training Loss (Raw)', color='blue', alpha=0.3)

                    # smoothed training loss using a moving average
                    smoothed_train = pd.Series(train_losses).rolling(window=30, min_periods=1).mean()
                    ax.plot(smoothed_train, label='Training Loss (Smoothed)', color='blue', linewidth=2)

                    # Plot validation loss
                    ax.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)

                    # Red dashed line at best epoch
                    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5,
                               label=f'Best Epoch: {best_epoch}')

                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Model {i + 1} Loss Curve')
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(f'model_{i + 1}_loss_epoch{epoch + 1}.png')
                    plt.close(fig)

                    break

        # Plot loss curves at specified epochs
        if (epoch + 1) % 100 == 0:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Plot raw training loss
            ax.plot(train_losses, label='Training Loss (Raw)', color='blue', alpha=0.3)

            # smoothed training loss using a moving average
            smoothed_train = pd.Series(train_losses).rolling(window=30, min_periods=1).mean()
            ax.plot(smoothed_train, label='Training Loss (Smoothed)', color='blue', linewidth=2)

            # Plot validation loss
            ax.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Model {i + 1} Loss Curve')
            ax.legend()
            fig.tight_layout()
            fig.savefig(f'model_{i + 1}_loss_epoch{epoch + 1}.png')
            plt.close(fig)



