import torch
import torch.nn as nn
import torch.optim as optim

class MicroscopyCNN(nn.Module):
    def __init__(self):
        super(MicroscopyCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjusted based on the size of the output after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single recurrence score

        # Dropout layers
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability to prevent overfitting

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the output for fully connected layers
        x = x.view(-1, 64 * 64 * 64)  # Adjust size based on the input image size (64x64 after pooling)

        # Apply dropout to the fully connected layers
        x = self.dropout(self.fc1(x))  # Dropout after fc1
        x = self.dropout(self.fc2(x))  # Dropout after fc2
        x = self.fc3(x)

        return x


# Initialize the model
model = MicroscopyCNN()

# Loss function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Optimizer (Adam with weight decay for L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization (weight decay)

# Print model summary
print(model)
