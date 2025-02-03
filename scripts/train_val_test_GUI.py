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
from tkinter import scrolledtext, Checkbutton, IntVar, Frame

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
batch_size = 16
epochs = 200
learning_rate = 1e-8

# Define transformations for training, validation, and test datasets
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)
])

val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)
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

# Initialize models
regression_model = MicroscopyCNN(task='regression')
classification_model = MicroscopyCNN(task='classification')

# Loss functions
regression_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()

# Optimizers
regression_optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate, weight_decay=0.01)
classification_optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate, weight_decay=0.01)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression_model.to(device)
classification_model.to(device)
print(f"Using device: {device}")

# GUI setup
root = tk.Tk()
root.title("Microscopy Model Training")

plot_frame = Frame(root)
plot_frame.pack()

fig_reg, ax_reg = plt.subplots(figsize=(4, 3))
ax_reg.set_xlabel('Epoch')
ax_reg.set_ylabel('Loss')
ax_reg.set_title('Regression Loss')
canvas_reg = FigureCanvasTkAgg(fig_reg, master=plot_frame)
canvas_reg.get_tk_widget().grid(row=0, column=0)

fig_class, ax_class = plt.subplots(figsize=(4, 3))
ax_class.set_xlabel('Epoch')
ax_class.set_ylabel('Loss')
ax_class.set_title('Classification Loss')
canvas_class = FigureCanvasTkAgg(fig_class, master=plot_frame)
canvas_class.get_tk_widget().grid(row=0, column=1)

log_box = scrolledtext.ScrolledText(root, width=50, height=10)
log_box.pack()

# Checkbuttons to select models for training
train_regression = IntVar()
train_classification = IntVar()
regression_check = Checkbutton(root, text="Train Regression Model", variable=train_regression)
classification_check = Checkbutton(root, text="Train Classification Model", variable=train_classification)
regression_check.pack()
classification_check.pack()


def train(model, criterion, optimizer, task, ax, canvas):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloaders['train'], desc=f"{task.upper()} Epoch {epoch + 1}/{epochs}"):
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

        ax.clear()
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.legend()
        canvas.draw()

        log_box.insert(tk.END,
                       f"{task.upper()} Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
        log_box.yview(tk.END)
        root.update()


def train_models():
    if train_regression.get():
        train(regression_model, regression_criterion, regression_optimizer, 'regression', ax_reg, canvas_reg)
    if train_classification.get():
        train(classification_model, classification_criterion, classification_optimizer, 'classification', ax_class,
              canvas_class)


def test_models():
    log_box.insert(tk.END, "Testing models...\n")
    log_box.yview(tk.END)


train_button = tk.Button(root, text="Train Selected Models", command=train_models)
train_button.pack()

test_button = tk.Button(root, text="Test Models", command=test_models)
test_button.pack()

root.mainloop()
