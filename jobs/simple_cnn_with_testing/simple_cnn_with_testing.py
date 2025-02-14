import torch
from torch.utils.data import DataLoader
from models.microscopy_cnn import MicroscopyCNN
from scripts.dataset_loader import MicroscopyDataset
import torch.optim as optim
import torch.nn as nn
import random
import torchvision.transforms.v2 as tvt
from scripts.model_metrics import score_model
import matplotlib.pyplot as plt
import pandas as pd

##------------------------------------------------------------##

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
batch_size = 16
epochs = 2500
learning_rate = 1e-6

# Transformations for training set
train_transform = tvt.Compose([
    tvt.RandomVerticalFlip(p=0.25),
    tvt.RandomHorizontalFlip(p=0.25),
    tvt.RandomRotation(degrees=(-180, 180))])

# Load dataset
train_dataset = MicroscopyDataset(
    csv_file="data/newData/labels.csv",
    root_dir="data/newData",
    transform=None
)

# Create a file to store the results
file = 'results.txt'
with open(file, 'w') as f:
    f.write('Results \n')

# Sample-based dataset splitting
samples_dict = train_dataset.samples  # Dictionary {sample_id: [list of FOVs]}
samples_list = list(samples_dict.items())  # Convert to list of tuples
print(f"Full sample list: {samples_list}")  # Print to output

with open(file, 'a') as f:
    f.write(f'Full sample list: {samples_list}\n')  # Write to results

random.shuffle(samples_list)  # Shuffle to avoid bias

# Compute split sizes
total_samples = len(samples_list)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size

# Split data based on sample_id
train_samples = samples_list[:train_size]
print(f"Training samples:{train_samples}")
with open(file, 'a') as f:
    f.write(f'Training samples: {train_samples}\n')  # Write to results

val_samples = samples_list[train_size:train_size + val_size]
print(f"Validation samples:{val_samples}")
with open(file, 'a') as f:
    f.write(f'Validation samples: {val_samples}\n')  # Write to results

test_samples = samples_list[train_size + val_size:]
print(f"Test samples:{test_samples}")
with open(file, 'a') as f:
    f.write(f'Test samples: {test_samples}\n')  # Write to results

# Function to flatten sample-wise FOVs into a dataset
def flatten_fovs(sample_list):
    return [fov for _, fovs in sample_list for fov in fovs]

# Assign FOVs to each dataset
train_dataset.samples = flatten_fovs(train_samples)

val_dataset = MicroscopyDataset(
    csv_file="data/newData/labels.csv",
    root_dir="data/newData",
    transform=None
)
val_dataset.samples = flatten_fovs(val_samples)

test_dataset = MicroscopyDataset(
    csv_file="data/newData/labels.csv",
    root_dir="data/newData",
    transform=None
)
test_dataset.samples = flatten_fovs(test_samples)


# Assign transformations to training dataset
train_dataset.transform = train_transform


# DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

# Initialize models
regression_model = MicroscopyCNN(task='regression')
classification_model = MicroscopyCNN(task='classification')

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

x = model(torch.rand(5,4,512,512))
print(x.shape)
print(x)

train_losses, val_losses = [], []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images).squeeze()
        if task == 'classification':
            labels = (labels > 30).float()

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
                labels = (labels > 30).float()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloaders['val'])
    val_losses.append(val_loss)

    if epoch == 0 or val_loss < best_loss:
        best_loss = val_loss
        torch.save(classification_model.state_dict(), "best_classification_model.pt")
        print(f'New best at epoch {epoch+1} with val loss {val_loss}')

        with open(file, 'a') as f:
            f.write(f'New best at epoch {epoch+1} with val loss {val_loss} \n')

    if (epoch+1)%250 == 0:
        # Save the trained model every 250 epochs
        torch.save(classification_model.state_dict(), f"classification_model_epoch{epoch+1}.pt")

        # Test the trained model every 250 epochs
        model.load_state_dict(torch.load(f"classification_model_epoch{epoch+1}.pt"))
        scores,fig = score_model(model, dataloaders['test'],print_results=True, make_plot=True, threshold_type='roc')
        fig.savefig(f'Epoch_{epoch+1}_test_plot.png')
        plt.close(fig)

        # Create training/val loss figure every 250 epochs
        ax = ax_class
        ax.clear()
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.legend()
        fig_class.savefig(f'loss_epoch{epoch}.png')

    # Print training/val loss every epoch
    print(f'Epoch{epoch + 1}: validation loss {val_loss}')
    print(f'Epoch{epoch + 1}: training loss {train_loss}')
    # and write them to "results.txt"
    with open(file, 'a') as f:
        f.write(f'Epoch {epoch+1}: val loss {val_loss} \n')
        f.write(f'Epoch {epoch+1}: train loss {train_loss} \n')




df = pd.DataFrame().assign(training_loss=train_losses,validation_loss=val_losses).to_csv('loss.csv', index=True)

