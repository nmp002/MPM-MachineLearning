## Imports ##
import torch
import os
from torch.utils.data import Subset
import torchvision.transforms.v2 as tvt
from torch.utils.data import DataLoader
from scripts.microscopy_dataset import MicroscopyDataset
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from models.multiclass_CNN import multiclassCNN
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

# ==================================
# PRESETS/PARAMETERS CONFIGURATION
# ==================================
channels = ['nadh', 'shg', 'orr']
in_channels = len(channels)
data_dir = "data/newData"
labels_csv = "data/newData/labels.csv"
label_fn = lambda x: torch.tensor(
    0 if 0 <= x <= 15 else   # Low Risk
    1 if 16 <= x <= 25 else  # Low to Medium Risk
    2 if 26 <= x <= 40 else  # Medium Risk
    3                        # High Risk
)

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
epochs = 500
learning_rate = 1e-6


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
# Split the sample_ids into train, val, and test sets
train_ids = ['Sample_020', 'Sample_025', 'Sample_004', 'Sample_026', 'Sample_014',
 'Sample_019', 'Sample_008', 'Sample_006', 'Sample_030', 'Sample_028',
 'Sample_003', 'Sample_022', 'Sample_002', 'Sample_027', 'Sample_013']


val_ids = ['Sample_012', 'Sample_018', 'Sample_024', 'Sample_029', 'Sample_017', 'Sample_007', 'Sample_010']


test_ids = ['Sample_011', 'Sample_016', 'Sample_009', 'Sample_015', 'Sample_023', 'Sample_005', 'Sample_001']


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

# =========================================
# INITIALIZE MODEL, OPTIMIZER, & LOSS FN
# =========================================
model = multiclassCNN(in_channels=in_channels)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# ==================================
# DEFINE SCORING FUNCTION
# ==================================
# Function to calculate scores and plot roc curve/confusion matrix
def score_em(t, o):
    true = np.array(t).astype(int)
    probs = softmax(o, axis=1)  # Apply softmax to model outputs

    # Confusion Matrix
    preds = np.argmax(probs, axis=1)
    conf_matrix = ConfusionMatrixDisplay.from_predictions(true, preds)
    conf_matrix.plot()
    plt.savefig(f'Epoch_{epoch + 1}_confusion_matrix.png')

    # ROC & AUC (per class)
    true_bin = label_binarize(true, classes=[0, 1, 2, 3])
    for i in range(4):
        fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title(f"ROC Curve - Class {i}")
        plt.savefig(f'Epoch_{epoch + 1}_roc_class_{i+1}.png')


# ==================================
# TRAINING LOOP
# ==================================
with open(results_file, 'a') as f:
    f.write(f'## Training and Validation:\n')

train_losses = []
val_losses = []
test_losses = []
best_loss = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x, target, _ in train_loader:
        x, target = x.to(device), target.to(device)
        target = target.long()
        optimizer.zero_grad()
        out = model(x)
        if (epoch+1) % 50 == 0:
            with open(results_file, 'a') as f:
                f.write(f'Output, Epoch_{epoch+1}: {out}\n')
                f.write(f'Target, Epoch_{epoch+1}: {target}\n')
        loss = loss_fn(out, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

# ==================================
# VALIDATION
# ==================================

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for x_val, target_val, _ in val_loader:
            x_val, target_val = x_val.to(device), target_val.to(device)
            target_val = target_val.long()
            out_val = model(x_val)
            val_loss = loss_fn(out_val, target_val)
            val_running_loss += val_loss.item()

    val_loss = val_running_loss / len(val_dataset)
    val_losses.append(val_loss)

# ==================================
# TESTING
# ==================================

    if (epoch+1) % 50==0:

        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.savefig(f'loss_plot_epoch_{epoch + 1}.png')
        plt.close()

        # Save the trained model every 250 epochs
        # torch.save(model.state_dict(), f"multiclass_model_epoch{epoch+1}.pt")
        with torch.no_grad():
            model.eval()
            ys, targets = [], []
            for img, target, _ in test_loader:
                img, target = img.to(device), target.to(device)
                target = target.long()
                y = model(img)
                y = y.cpu().numpy()
                # y = y.numpy().astype(np.float64).tolist()
                target = target.cpu().numpy()
                # target = target.numpy().astype(np.float64).tolist()
                ys.append(y)
                targets.append(target)

            ys = np.concatenate(ys, axis=0)
            targets = np.concatenate(targets)

            num_samples = len(ys) // 5
            ys_reshaped = ys.reshape(num_samples, 5, 4)
            targets_reshaped = targets.reshape(num_samples, 5)

            # Average predictions and targets across FOVs
            sample_preds = np.mean(ys_reshaped, axis=1)
            sample_targets = np.mean(targets_reshaped, axis=1)
            sample_targets = sample_targets.astype(int)

            # Pass through scoring function
            score_em(sample_targets, sample_preds)

            # Old method for reference:
            # ys = [item for y in ys for item in y]
            # sample_ys = np.mean(np.array(ys).reshape(-1, 5), axis=1)
            # targets = [item for target in targets for item in target]
            # sample_targets = np.mean(np.array(targets).reshape(-1, 5), axis=1)
            # score_em(sample_targets, sample_ys)



