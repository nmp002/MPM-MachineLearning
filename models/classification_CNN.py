####################### Description #######################
# This is an updated version of my old MicroscopyCNN model
# Main updates:
#   Handles classification only (not regression)
#   ReLUs added between each fc layer
#############################################################

import torch.nn as nn

class classificationModel(nn.Module):
    def __init__(self, in_channels=4):
        """
        in_channels: number of input channels (e.g., 1 for FAD only, 2 for FAD+NADH, etc.)
        task: 'regression' or 'classification'
        """
        super(classificationModel, self).__init__()
        # Statics
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjusted based on the size of the output after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1)
        x = self.drop(x)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.flat(x)  # Adjust size based on the input image size (64x64 after pooling)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        x = self.sigmoid(x)
        return x