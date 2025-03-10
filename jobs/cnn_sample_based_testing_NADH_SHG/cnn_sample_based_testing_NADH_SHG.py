import torch
from torch.utils.data import DataLoader
from models.microscopy_cnn import MicroscopyCNN
from scripts.dataset_loader import MicroscopyDataset
import torch.optim as optim
import torch.nn as nn
import random
import torchvision.transforms.v2 as tvt
from sklearn.metrics import roc_auc_score
from scripts.model_metrics import score_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
from scripts.dataset_loader import tiff_to_tensor
import numpy as np

##------------------------------------------------------------##

plt.close('all')

# Choose which channels to use for input
input_channels = ['nadh','shg'] # Options: 'fad', 'nadh', 'shg', 'orr'
in_channels = len(input_channels)

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(40) # changed from 42 to 40

# Hyperparameters
batch_size = 16
epochs = 500
learning_rate = 1e-4

# Transformations for training set
train_transform = tvt.Compose([
    tvt.RandomVerticalFlip(p=0.25),
    tvt.RandomHorizontalFlip(p=0.25),
    tvt.RandomRotation(degrees=(-180, 180))])


# Load full dataset
full_dataset = MicroscopyDataset(
    csv_file="data/newData/labels.csv",
    root_dir="data/newData",
    channels = input_channels,
    transform=None
)

train_dataset = MicroscopyDataset(
    csv_file="data/newData/labels.csv",
    root_dir="data/newData",
    channels = input_channels,
    transform=None
)
# Create a file to store the results
file = 'results.md'
with open(file, 'w') as f:
    f.write('**Results** \n\n')

train_dataset.transform = train_transform


# Compute split sizes
total_samples = len(full_dataset.sample_wise_paths)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size



indices = torch.utils.data.SubsetRandomSampler(range(total_samples))
indices = [i for i in indices]

# Split data based on sample_id
# train_samples = indices[:train_size]
train_samples = [18, 14, 10, 22, 21, 6, 5, 12, 11, 15, 9, 24, 19, 28, 1, 13, 2, 17]
print(f"Train samples: {train_samples}")
print(f"Training samples:")
for id_num in train_samples:
    if id_num >=20:
        print(f"Sample_{(id_num+2):03}")
    else:
        print(f"Sample_{(id_num+1):03}")
with open(file, 'a') as f:
    f.write('**Training samples:**')
    for id_num in train_samples:
        if id_num >= 20:
            f.write(f" Sample_{(id_num+2):03} |")  # Write which samples are used in training to results
        else:
            f.write(f" Sample_{(id_num+1):03} |")
    f.write('\n\n')

# val_samples = indices[train_size:train_size + val_size]
val_samples = [27, 4, 25, 7, 8]
print(f"Validation samples:")
for id_num in val_samples:
    if id_num >= 20:
        print(f"Sample_{(id_num+2):03}")
    else:
        print(f"Sample_{(id_num+1):03}")
with open(file, 'a') as f:
    f.write('**Validation samples:**')
    for id_num in val_samples:
        if id_num >= 20:
            f.write(f" Sample_{(id_num+2):03} |")  # samples in validation
        else:
            f.write(f" Sample_{(id_num+1):03} |")
    f.write('\n\n')

# test_samples = indices[train_size + val_size:]
test_samples = [23, 0, 3, 20]
print(f"Test samples:")
for id_num in test_samples:
    if id_num >= 20:
        print(f"Sample_{(id_num+2):03}")
    else:
        print(f"Sample_{(id_num+1):03}")
with open(file, 'a') as f:
    f.write('**Test samples**:')
    for id_num in test_samples:
        if id_num >= 20:
            f.write(f' Sample_{(id_num+2):03} |')  # samples in testing
        else:
            f.write(f" Sample_{(id_num+1):03} |")
    f.write(f'\n\n{"-" * 100} \n\n')


train_indices = [full_dataset.get_sample_indices(sample) for sample in train_samples]
train_indices = [i for sublist in train_indices for i in sublist]
train_data = torch.utils.data.Subset(train_dataset, train_indices)

val_indices = [full_dataset.get_sample_indices(sample) for sample in val_samples]
val_indices = [i for sublist in val_indices for i in sublist]
val_data = torch.utils.data.Subset(full_dataset, val_indices)

test_indices = [full_dataset.get_sample_indices(sample) for sample in test_samples]
test_indices = [i for sublist in test_indices for i in sublist]
test_data = torch.utils.data.Subset(full_dataset, test_indices)


# DataLoaders
dataloaders = {
    'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_data, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_data, batch_size=len(test_data), shuffle=False)
}

# Initialize models
regression_model = MicroscopyCNN(in_channels=in_channels, task='regression')
classification_model = MicroscopyCNN(in_channels=in_channels, task='classification')

# Loss functions
regression_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression_model.to(device)
classification_model.to(device)
print(f"Using device: {device}")

# Optimizers
regression_optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate, weight_decay=0.01)
classification_optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate, weight_decay=0.01)


fig_class, ax_class = plt.subplots(figsize=(4, 3))
ax_class.set_xlabel('Epoch')
ax_class.set_ylabel('Loss')
ax_class.set_title('Classification Loss')


# Define models for training
train_regression = 0
train_classification = 1



model = classification_model
optimizer = classification_optimizer
criterion = classification_criterion
task = 'classification'

# Function to calculate scores and plot roc curve/confusion matrix
def score_em(t, o):
    fpr, tpr, thresholds = roc_curve(t, o)
    test_score = auc(fpr, tpr)
    thresh = thresholds[np.argmax(tpr - fpr)]
    preds = [out >= thresh for out in o]
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=test_score)
    roc_display.plot()
    plt.savefig(f'Epoch_{epoch + 1}_roc_curve.png')
    conf_matrix = ConfusionMatrixDisplay.from_predictions(t, preds)
    conf_matrix.plot()
    plt.savefig(f'Epoch_{epoch + 1}_confusion_matrix.png')

train_losses, val_losses = [], []
best_loss = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images).squeeze()
        if task == 'classification':
            labels = (labels > 25).float()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(dataloaders['train'])
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            if task == 'classification':
                labels = (labels > 25).float()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloaders['val'])
    val_losses.append(val_loss)

    if epoch == 0 or val_loss < best_loss:
        best_loss = val_loss
        torch.save(classification_model.state_dict(), "best_classification_model.pt")
        print(f'New best at epoch {epoch+1} with val loss {val_loss}')

        with open(file, 'a') as f:
            f.write(f'New best at epoch **{epoch+1}** with val loss **{val_loss}** \n')

    if (epoch+1)%250 == 0:
        # Save the trained model every 250 epochs
        torch.save(classification_model.state_dict(), f"classification_model_epoch{epoch+1}.pt")

        # Test the trained model every 250 epochs
        model.load_state_dict(torch.load(f"classification_model_epoch{epoch+1}.pt"))

        with torch.no_grad():
            model.eval()

            for img, targets in dataloaders['test']:
                img = img.to(device)
                y = model(img).cpu().squeeze()

        targets = [1 if t > 25 else 0 for t in targets]
        y = y.numpy().astype(np.float64).tolist()
        score_em(targets, y)


        # Create training/val loss figure every 250 epochs
        ax = ax_class
        ax.clear()
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.legend()
        fig_class.savefig(f'loss_epoch{epoch+1}.png')

    # Print training/val loss every epoch
    print(f'Epoch{epoch + 1}: validation loss {val_loss}')
    print(f'Epoch{epoch + 1}: training loss {train_loss}')
    # and write them to "results.md"
    with open(file, 'a') as f:
        f.write(f'Epoch {epoch+1}: **val loss {val_loss}** \n')
        f.write(f'Epoch {epoch+1}: **train loss {train_loss}** \n')




df = pd.DataFrame().assign(training_loss=train_losses,validation_loss=val_losses).to_csv('loss.csv', index=True)