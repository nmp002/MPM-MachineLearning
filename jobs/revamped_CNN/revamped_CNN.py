## Imports ##
import torch
from sklearn.model_selection import train_test_split
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
epochs = 2500
learning_rate = 1e-6

# SET SPLITS
val_split = 0.17
test_split = 0.13

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
    f.write(f'### Channels used: {channels}\n')
    f.write(f'### Batch size: {batch_size}\n')
    f.write(f'### Epochs: {epochs}\n')
    f.write(f'### Learning rate: {learning_rate}\n')

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
# IMAGE CHECKPOINT 1
# ==================================
# with open(results_file, 'a') as f:
#     f.write('## Image Checkpoint 1:\n')
#
# # Plot 5 random samples to ensure images are loaded correctly from dataset
# total_samples = len(dataset)
# random_indices = random.sample(range(total_samples), 5)
# for i, idx in enumerate(random_indices):
#     image, label, sample_id = dataset[idx]
#     fig, ax = plt.subplots(1, in_channels)
#     if in_channels == 1:
#         ax = [ax]
#     for channel in range(in_channels):
#         channel_image = image[channel].numpy()
#         ax[channel].imshow(channel_image, cmap='gray')
#         ax[channel].set_title(f'Channel {channel+1} - Sample: {sample_id}\nLabel: {label.item()}')
#         ax[channel].axis('off')
#     plt.tight_layout()
#
#     plot_filename = f'{sample_id}_Index_{idx}.png'
#     plt.savefig(plot_filename)
#     plt.close(fig)
#
#     with open(results_file, 'a') as f:
#         f.write(f'## {sample_id} (Label: {label.item()})\n\n')
#         f.write(f'![{sample_id}]({plot_filename})\n\n')

    # if os.path.exists(plot_filename):
    #     os.remove(plot_filename)
    # else:
    #     print(f'Warning: {plot_filename} does not exist')

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
sample_ids = list(set([sample_id for _, _, _, sample_id in dataset.img_labels]))

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

# =========================================
# INITIALIZE MODEL, OPTIMIZER, & LOSS FN
# =========================================
model = classificationModel(in_channels=in_channels)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_fn = nn.BCELoss()



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
        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = loss_fn(out, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

#dlhfd
# ==================================
# VALIDATION AND TESTING
# ==================================
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, target, _ in val_loader:
            x, target = x.to(device), target.to(device)
            out = model(x).squeeze()
            loss = loss_fn(out, target)
            running_loss += loss.item()

        val_loss = running_loss / len(val_dataset)
        val_losses.append(val_loss)

    if epoch == 0 or val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        with open(results_file, 'a') as f:
            f.write(f'New best at epoch **{epoch+1}** with val loss **{val_loss}** \n')

    with open(results_file, 'a') as f:
        f.write(f'Epoch {epoch+1}: **val loss {val_loss}** \n')
        f.write(f'Epoch {epoch+1}: **train loss {train_loss}** \n')



    if (epoch+1) % 250 == 0:
        running_loss = 0.0
        for x, target, _ in test_loader:
            x, target = x.to(device), target.to(device)
            out = model(x).squeeze()
            loss = loss_fn(out, target)
            running_loss += loss.item()
            # score = score_em(out, target)

        test_loss = running_loss / len(test_dataset)
        test_losses.append(test_loss)

        # Save as desired

        # Create training/val loss figure every 250 epochs
        fig_class, ax_class = plt.subplots(figsize=(4, 3))
        ax_class.set_xlabel('Epoch')
        ax_class.set_ylabel('Loss')
        ax_class.set_title('Loss Curves')

        ax = ax_class
        ax.clear()
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.legend()
        fig_class.savefig(f'loss_epoch{epoch+1}.png')


