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
epochs = 100
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load dataset with training transformations
dataset = MicroscopyDataset(
    csv_file="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data/labels.csv",
    root_dir="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data",
    transform=None  # No transform applied directly here; will assign per split below
)

# Split dataset into training (70%), validation (20%), and test (10%)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Assign transformations to datasets
train_dataset.dataset.transform = train_transform  # Apply data augmentation for training
val_dataset.dataset.transform = val_test_transform  # No augmentation for validation
test_dataset.dataset.transform = val_test_transform  # No augmentation for testing

# DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

# Initialize the model, loss function, and optimizer
model = MicroscopyCNN()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# GUI setup
root = tk.Tk()
root.title("Microscopy Model Training")

# Add a plot for training and validation loss
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.set_xlim(0, epochs)
ax.set_ylim(0, 1)

canvas = FigureCanvasTkAgg(fig, master=root)  # A Tkinter canvas
canvas.get_tk_widget().pack()

# Text box for logs
log_box = scrolledtext.ScrolledText(root, width=50, height=10)
log_box.pack()

# Training function with live updates to the GUI
def train():
    train_losses = []
    val_losses = []  # Track validation losses

    def update_plot():
        ax.clear()  # Clear the previous plot
        ax.plot(train_losses, label='Training Loss', color='blue')  # Plot training loss
        ax.plot(val_losses, label='Validation Loss', color='orange')  # Plot validation loss
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlim(0, max(len(train_losses), len(val_losses)))  # Adjust x-axis dynamically
        all_losses = train_losses + val_losses
        ax.set_ylim(0, max(all_losses) + 0.1)  # Adjust y-axis based on both losses
        ax.legend()
        canvas.draw()

    for epoch in range(epochs):
        # Training step
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

        # Validation step
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

        # Log the losses
        log_box.insert(tk.END, f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {train_loss:.4f}\n Validation Loss: {val_loss:.4f}\n")
        log_box.yview(tk.END)  # Scroll to the bottom

        # Update the plot
        update_plot()

        # Manually update the GUI
        root.update()

    if input("Save trained model? (yes/no): ").strip().lower() == 'yes':
        torch.save(model.state_dict(), "../models/microscopy_model.pth")
        log_box.insert(tk.END, "Model saved.\n")
        log_box.yview(tk.END)

# Function to test the model
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

# Buttons to control the operations
train_button = tk.Button(root, text="Train", command=train)
train_button.pack()

test_button = tk.Button(root, text="Test", command=test)
test_button.pack()

# Run the Tkinter event loop
root.mainloop()
