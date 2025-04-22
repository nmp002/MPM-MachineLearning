######################## Description ############################
# Convolutional Neural Network for Classification of Breast
# Cancer Biopsy Samples by Oncotype DX Recurrence Score Class
#
# Created by Nicholas Powell
# Laboratory for Functional Optical Imaging & Spectroscopy
# University of Arkansas
#
# Please note: For easier reference, images are referred to
# as NADH, FAD, SHG, and ORR instead of I_755/blue, I_855/green,
# I_855/UV, and optical ratio, respectively.
#################################################################

import torch.nn as nn


class classificationModel(nn.Module):
    def __init__(self, in_channels=4):
        """
        Initialize the classification model.

        Parameters:
        - in_channels (int): Number of input channels
          (e.g., 1 for FAD only, 2 for FAD+NADH, etc.).
          Use relevant number of input channels based on your application.

        The model consists of:
        - 3 convolutional layers for feature extraction.
        - MaxPooling layers for spatial dimension reduction.
        - ReLU activations after each layer for non-linearity.
        - Dropout for regularization.
        - Fully Connected (FC) layers to perform the final classification.
        """
        super(classificationModel, self).__init__()

        # Non-linear activation function used throughout the network.
        self.relu = nn.ReLU()

        # Max-pooling layer for down-sampling the feature maps.
        self.pool = nn.MaxPool2d(2, 2)

        # Flattens the multi-dimensional feature maps to a 1D vector for the fully connected layers.
        self.flat = nn.Flatten()

        # Dropout layer to prevent overfitting. Dropout rate is set to 0.5.
        self.drop = nn.Dropout(0.5)

        # Sigmoid activation for binary classification (output probability between 0 and 1).
        self.sigmoid = nn.Sigmoid()

        # First convolutional layer: Detects 16 features (filters) from the input.
        # Each filter operates over a 3x3 area of the input. Padding ensures output dimensions match input dimensions.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer: Increases feature set to 32 filters.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Third convolutional layer: Increases feature set to 64 filters.
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected (FC) layers:
        # Connect flattened features to classification nodes.

        # FC1: Input size is determined by flattening the pooled features (after 3 pooling operations).
        # Multiplication of input size here assumes image size is 64x64 after pooling.
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Outputs 128 nodes.

        # FC2: Further reduce to 64 nodes.
        self.fc2 = nn.Linear(128, 64)

        # FC3: Final output is 1 node, representing the binary classification output.
        self.fc3 = nn.Linear(64, 1)  # Outputs a single value (probability).

    def forward(self, x):
        """
        Define the forward pass through the network.

        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
        - Tensor: Output tensor with predicted probabilities (shape: [batch_size, 1]).
        """
        # Pass input through the first convolutional layer.
        x = self.conv1(x)  # Convolutional Layer 1
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max-pooling layer

        # Pass input through the second convolutional layer.
        x = self.conv2(x)  # Convolutional Layer 2
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max-pooling layer

        # Pass input through the third convolutional layer.
        x = self.conv3(x)  # Convolutional Layer 3
        x = self.relu(x)  # ReLU activation
        x = self.pool(x)  # Max-pooling layer

        # Apply dropout for regularization right after the convolutional layers.
        x = self.drop(x)

        # Flatten multi-dimensional output to a single-dimensional tensor for FC layers.
        x = self.flat(x)  # Adjust size based on the input image size.

        # Pass through first fully connected layer.
        x = self.fc1(x)  # Fully Connected Layer 1
        x = self.relu(x)  # ReLU activation
        x = self.drop(x)  # Apply dropout for regularization

        # Pass through second fully connected layer.
        x = self.fc2(x)  # Fully Connected Layer 2
        x = self.relu(x)  # ReLU activation
        x = self.drop(x)  # Apply dropout for further regularization

        # Pass through the final fully connected layer for classification.
        x = self.fc3(x)  # Fully Connected Layer 3

        # Apply sigmoid activation to generate output probabilities.
        x = self.sigmoid(x)

        # Return the final output tensor with probabilities.
        return x
