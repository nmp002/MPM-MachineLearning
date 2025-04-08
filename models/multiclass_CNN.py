####################### Description #######################
# This model handles multiclass classification (more than 2 Recurrence Score Ranges --> 5 total)
# Main updates:
#   Final layer outputs 5 logits instead of 1
#   Sigmoid removed
# Created on: March 18th, 2025
#############################################################

import torch.nn as nn

class multiclassCNN(nn.Module):
    def __init__(self, in_channels=4):
        """
        in_channels: number of input channels (e.g., 1 for FAD only, 2 for FAD+NADH, etc.)
        task: 'regression' or 'classification'
        """
        super(multiclassCNN, self).__init__()
        # Statics
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjusted based on the size of the output after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Multi-class classification output

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


        x = self.drop(x)

        x = self.flat(x)  # Adjust size based on the input image size (64x64 after pooling)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        # x = self.sigmoid(x)
        return x