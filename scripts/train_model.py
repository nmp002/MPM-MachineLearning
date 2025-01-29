import torch
from torch.utils.data import DataLoader, random_split
from models.microscopy_cnn import MicroscopyCNN
from dataset_loader import MicroscopyDataset
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
batch_size = 16
epochs = 200
learning_rate = 1e-4

# Define transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 or another desired size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize with mean and std
])

# Load dataset
dataset = MicroscopyDataset(csv_file="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data/labels.csv",
                            root_dir="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/data",
                            transform=transform)

# Split dataset into training (70%), validation (20%), and test (10%)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

# Initialize the model, loss function, and optimizer
model = MicroscopyCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# Function to train the model
def train():
    train_losses = []
    plt.ion()
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
        epoch_loss = running_loss / len(dataloaders['train'])
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {epoch_loss:.4f}")

        # Live update of the training loss plot
        plt.clf()
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.pause(0.1)

        # Validate the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            validate()

    plt.ioff()
    plt.show()

    if input("Save trained model? (yes/no): ").strip().lower() == 'yes':
        torch.save(model.state_dict(), "../models/microscopy_model.pth")
        print("Model saved.")
    print("Training complete.")


# Function to load the model
def load_model(
        model_path="C:/Users/nmp002/PycharmProjects/HighlandsMachineLearning/models/27Jan2025/microscopy_model.pth"):
    model = MicroscopyCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


# Function to validate the model
def validate():
    model = load_model()  # Load the saved model
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Debugging outputs and labels shapes
            print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging output

            # Print predictions and labels
            print(
                f"Predictions: {outputs.squeeze().cpu().numpy()}, Labels: {labels.cpu().numpy()}")  # Print sample predictions

            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(dataloaders['val']):.4f}")


# Function to test the model
def test():
    model = load_model()  # Load the saved model
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Debugging outputs and labels shapes
            print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging output

            # Print predictions and labels
            print(
                f"Predictions: {outputs.squeeze().cpu().numpy()}, Labels: {labels.cpu().numpy()}")  # Print sample predictions

            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(dataloaders['test']):.4f}")


# User prompt to choose operation
if __name__ == "__main__":
    choice = input("Enter operation (train/validate/test): ").strip().lower()
    if choice == 'train':
        train()
    elif choice == 'tr':
        train()
    elif choice == 'validate':
        validate()  # For validation
    elif choice == 'v':
        validate()
    elif choice == 'test':
        test()  # For testing
    elif choice == 't':
        test()
    else:
        print("Invalid choice. Please enter 'train (tr)', 'validate (v)', or 'test (t)'.")
