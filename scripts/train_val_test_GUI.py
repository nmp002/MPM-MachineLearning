import torch
from torch.utils.data import DataLoader, random_split
from models.microscopy_cnn import MicroscopyCNN
from dataset_loader import MicroscopyDataset
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import scrolledtext

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
batch_size = 16
epochs = 200
learning_rate = 1e-4

# Define transformations for training, validation, and test datasets
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
])

# Load dataset
dataset = MicroscopyDataset(
    csv_file="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data/labels.csv",
    root_dir="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data",
    transform=None
)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Assign transformations to datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

# DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

# Initialize model, loss function, optimizer, and scheduler
model = MicroscopyCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce LR every 20 epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

for images, labels in dataloaders['train']:
    print("Image shape:", images.shape)  # Should be [batch_size, 4, 512, 512]
    print("Label shape:", labels.shape)  # Should be [batch_size]
    break



# GUI setup
root = tk.Tk()
root.title("Microscopy Model Training")

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.set_xlim(0, epochs)
ax.set_ylim(0, 1)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

log_box = scrolledtext.ScrolledText(root, width=50, height=10)
log_box.pack()

def train():
    train_losses = []
    val_losses = []

    def update_plot():
        ax.clear()
        ax.plot(train_losses, label='Training Loss', color='blue')
        ax.plot(val_losses, label='Validation Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlim(0, max(len(train_losses), len(val_losses)))
        ax.set_ylim(0, max(train_losses + val_losses) + 0.1)
        ax.legend()
        canvas.draw()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
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
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        val_loss = val_loss / len(dataloaders['val'])
        val_losses.append(val_loss)

        scheduler.step()  # Step the learning rate scheduler

        log_box.insert(tk.END, f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}\n")
        log_box.yview(tk.END)
        update_plot()
        root.update()

def test():
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
    log_box.insert(tk.END, f"Test Loss: {test_loss / len(dataloaders['test']):.4f}\n")
    log_box.yview(tk.END)

train_button = tk.Button(root, text="Train", command=train)
train_button.pack()

test_button = tk.Button(root, text="Test", command=test)
test_button.pack()

root.mainloop()
