######################## Description ############################
# Method 1 Script Used to Train the CNN on All Four Image Inputs,
# Using a Static Train/Val/Test Split Approach
#
# Created by Nicholas Powell
# Laboratory for Functional Optical Imaging & Spectroscopy
# University of Arkansas
#
# Please note: For easier reference, images are referred to
# as NADH, FAD, SHG, and ORR instead of I_755/blue, I_855/green,
# I_855/UV, and optical ratio, respectively.
#################################################################

## Imports ##
import pandas as pd
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
# Setting configuration paths for input data and labels
data_dir = "data/newData"   # Path to data directory containing sample imagees
labels_csv = "data/newData/labels.csv"  # Path to CSV file containing Recurrence Scores
# Lambda function to assign a binary label based on a threshold:
# True if RS>25, else False
label_fn = lambda x: torch.tensor(float(x > 25))

# Function to set random seed for reproducibility across libraries and experiments
def set_seed(seed: int = 42) -> None:
    """
    Set a fixed random seed for all libraries and configurations
    to ensure reproducibility of experiments.
    Parameters:
    - seed (int): The random seed value. Default is 42.
    """
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
# All can be adjusted to monitor and fine-tune model performance
batch_size = 16 # Number of samples per training batch
epochs = 3000   # Max number of epochs to train
learning_rate = 1e-6  # Learning rate for optimization
weight_decay = 0.01 # Regularization parameter to prevent overfitting -- changed for Iterations 1-4

# Automatically switch between GPU (if available) and CPU
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
# Define and initialize a results file to log progress
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
def score_em(t, o):
    """
    Calculates evaluation metrics (ROC curve, AUC, thresholds), and saves
    plots for the ROC curve and confusion matrix.
    Parameters:
    - t: Ground truth labels (targets)
    - o: Model predictions (probabilities/outputs)
    """
    # Compute false-positive and true-positive rates and thresholds
    fpr, tpr, thresholds = roc_curve(t, o)
    test_score = auc(fpr, tpr)  # Calculate the AUC score
    # Determine the threshold that maximizes TPR - FPR (Youden J Statistic)
    thresh = thresholds[np.argmax(tpr - fpr)]
    with open(results_file, 'a') as f:
        f.write(f'Threshold = {thresh}')
    # Generate binary predictions using the selected threshold
    preds = [out_value >= thresh for out_value in o]
    # Plot and save the ROC curve
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=test_score)
    roc_display.plot()
    plt.savefig(f'Model_{i+1}_epoch{best_epoch + 1}_roc_curve.png')
    # Plot and save the confusion matrix
    conf_matrix = ConfusionMatrixDisplay.from_predictions(t, preds)
    conf_matrix.plot()
    plt.savefig(f'Model_{i+1}_epoch{best_epoch + 1}_confusion_matrix.png')


# Function to get indices of img_labels belonging to a given set of sample_ids
def get_indices_by_sample_ids(img_labels, sample_ids_set):
    """
    Retrieve indices of data samples corresponding to given sample IDs.
    Parameters:
    - img_labels: List of image labels from the dataset.
    - sample_ids_set: Set containing the target sample IDs.
    Returns:
    - indices: List of indices corresponding to the specified sample IDs.
    """
    indices = []
    for idx, (sample_dir, fov_dir, label, sample_id) in enumerate(img_labels):
        if sample_id in sample_ids_set: # Check if sample ID matches
            indices.append(idx)
    return indices

# =========================================
# INITIALIZE MODELS, OPTIMIZERS, & LOSS FNS
# =========================================
# Configured so that multiple channel sets and models
# can be trained/evaluated at once
channel_set = [['nadh', 'fad', 'shg', 'orr']]     # Add more in list format if desired

# Empty lists for dynamic initialization of models, optimizers, datasets, loaders, etc.
models, optimizers, loss_fns = [], [], []
datasets, train_datasets = [], []
train_loaders, val_loaders, test_loaders = [], [], []

# Sample distribution for training (20), validation (5), and testing (4) sets.
# Sets were pre-chosen randomly outside of this script
train_ids = ['Sample_019', 'Sample_015', 'Sample_011', 'Sample_024', 'Sample_005', 'Sample_007', 'Sample_006', 'Sample_022', 'Sample_009', 'Sample_016', 'Sample_010',
             'Sample_025', 'Sample_028', 'Sample_030', 'Sample_017', 'Sample_029', 'Sample_027', 'Sample_014', 'Sample_003', 'Sample_018']

val_ids = ['Sample_020', 'Sample_023', 'Sample_002', 'Sample_008', 'Sample_012']

test_ids = ['Sample_026', 'Sample_001', 'Sample_004', 'Sample_013']

# Main loop to initialize a model for each channel set
for channels in channel_set:
    in_channels = len(channels)
    # Initialize CNN model for binary classification with the given input channels
    model = classificationModel(in_channels=in_channels).to(device)
    # Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Binary Cross-Entropy Loss for training a binary classifier
    loss_fn = nn.BCELoss()

    # Create dataset with appropriate arguments
    dataset = MicroscopyDataset(
        data_dir=data_dir,
        labels_csv=labels_csv,
        channels=channels,
        transform=None,
        label_fn=label_fn
    )

    # Create a separate training dataset and apply training transforms
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
    test_indices = get_indices_by_sample_ids(dataset.img_labels, set(test_ids))

    # Create dataset subsets using the computed indices
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create data loaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Append these elements to their respective lists (for handling multiple models)
    models.append(model)
    optimizers.append(optimizer)
    loss_fns.append(loss_fn)
    datasets.append(dataset)
    train_datasets.append(train_dataset)
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)
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
        if (epoch+1) == 1 or (epoch+1) % 25 == 0:
            with open(results_file, 'a') as f:
                f.write(f'Epoch {epoch+1}/{epochs}\n')

        model = models[i]
        optimizer = optimizers[i]
        loss_fn = loss_fns[i]

        # Train the model
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
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
            num_train_batches += 1
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        ## Validation ##
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for x, target, _ in val_loaders[i]:
                x, target = x.to(device), target.to(device)
                out = model(x).squeeze()
                loss = loss_fn(out, target)
                epoch_val_loss += loss.item()
                num_val_batches += 1

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
                    f.write(f"🔁 Epoch {epoch + 1}: No improvement in validation loss. Patience counter: {patience_counter}/{patience}\n")
                if patience_counter >= patience:
                    with open(results_file, 'a') as f:
                        f.write(f"⏹️ Early stopping triggered at Epoch {epoch + 1}.\n")

                        f.write(f'\nTesting Model_{i+1}:\n')

                    early_stopping = True
                    model.load_state_dict(torch.load(best_model_path))

                    ## Testing ##
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
                        t = list(averaged_targets.values())
                        threshold = 0.5
                        # Generate binary predictions using the selected threshold
                        preds = [out_value >= threshold for out_value in sample_ys]

                        # Plot and save the confusion matrix
                        conf_matrix = ConfusionMatrixDisplay.from_predictions(t, preds)
                        conf_matrix.plot()
                        plt.savefig(f'Model_{i + 1}_epoch{best_epoch + 1}_confusion_matrix.png')


                    fig, ax = plt.subplots(figsize=(6, 4))

                    # Plot raw training loss
                    ax.plot(train_losses, label='Training Loss (Raw)', color='blue', alpha=0.3)

                    # smoothed training loss using a moving average
                    smoothed_train = pd.Series(train_losses).rolling(window=30, min_periods=1).mean()
                    ax.plot(smoothed_train, label='Training Loss (Smoothed)', color='blue', linewidth=2)

                    # Plot validation loss
                    ax.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)

                    # Red dashed line at best epoch
                    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, label = f'Best Epoch: {best_epoch+1}')

                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Model {i + 1} Loss Curve')
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(f'model_{i + 1}_loss_epoch{epoch + 1}.png')
                    plt.close(fig)

                    break

        # Plot loss curves at specified epochs
        if (epoch+1) % 50 ==0 or early_stopping:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Plot raw training loss
            ax.plot(train_losses, label='Training Loss (Raw)', color='blue', alpha = 0.3)

            # smoothed training loss using a moving average
            smoothed_train = pd.Series(train_losses).rolling(window=30, min_periods=1).mean()
            ax.plot(smoothed_train, label='Training Loss (Smoothed)', color='blue', linewidth=2)

            # Plot validation loss
            ax.plot(val_losses, label='Validation Loss', color = 'orange', linewidth = 2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Model {i + 1} Loss Curve')
            ax.legend()
            fig.tight_layout()
            fig.savefig(f'model_{i + 1}_loss_epoch{epoch + 1}.png')
            plt.close(fig)





