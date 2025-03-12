import torch
import torch.nn as nn
import torch.optim as optim


class MicroscopyCNN(nn.Module):
    def __init__(self, in_channels=4, task='regression'):
        """
        in_channels: number of input channels (e.g., 1 for FAD only, 2 for FAD+NADH, etc.)
        task: 'regression' or 'classification'
        """
        super(MicroscopyCNN, self).__init__()
        self.task = task

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
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

        if self.task == 'regression':
            self.fc3 = nn.Linear(64, 1)  # Single output for regression
        else:
            self.fc3 = nn.Linear(64, 1)  # Binary classification output

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

        if self.task == 'regression':
            return self.fc3(x)  # Raw output for regression
        else:
            return torch.sigmoid(self.fc3(x))  # Sigmoid for classification


# Initialize models
regression_model = MicroscopyCNN(task='regression')
classification_model = MicroscopyCNN(task='classification')

# Loss function
regression_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()

# Optimizer (Adam with weight decay for L2 regularization)
regression_optimizer = optim.Adam(regression_model.parameters(), lr=1e-3, weight_decay=0.01)  # L2 regularization
classification_optimizer = optim.Adam(classification_model.parameters(), lr=1e-3, weight_decay=0.01)

# Print model summary
# print(regression_model)
# print(classification_model)
